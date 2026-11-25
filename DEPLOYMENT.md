# Production Deployment Guide: Vehicle Maintenance AI Agent on AWS EC2

This guide outlines the complete end-to-end procedure for deploying the **Vehicle Maintenance AI Agent** on AWS EC2 using Docker and Docker Compose.

---

## 1. System Architecture Overview

The application is containerized into four coordinated services managed by Docker Compose on an isolated internal bridge network:

```
                          [ Internet / User Browser ]
                                       │
                                       ▼ (Port 80 / 443)
                  ┌──────────────────────────────────────────┐
                  │          vm_frontend (Nginx)             │
                  │   - Serves React SPA static assets       │
                  │   - Reverse-proxies /api/ and /auth/     │
                  │   - Handles SSE stream unbuffered        │
                  │   - Gateway for Swagger /docs            │
                  └────────────────────┬─────────────────────┘
                                       │ (Internal Docker Network: vm_network)
                                       ▼ (Port 7860)
                  ┌──────────────────────────────────────────┐
                  │          vm_backend (FastAPI)            │
                  │   - Google ADK Multi-Agent Orchestrator  │
                  │   - Pure ML Random Forest Predictor      │
                  │   - Local SHAP Explainability Engine     │
                  │   - LLM Router (Groq/OpenAI failover)    │
                  └────────────┬───────────────────┬─────────┘
                               │                   │
                               ▼ (Port 5432)       ▼ (Port 6379)
             ┌─────────────────────────┐   ┌────────────────────────┐
             │  vm_postgres (Postgres) │   │    vm_redis (Redis)    │
             │  - Conversations & auth │   │  - In-memory cache     │
             │  - Persistent volume    │   │  - Session speedup     │
             └─────────────────────────┘   └────────────────────────┘
```

### Key Security & Isolation Features
- **Single Public Endpoint**: Only Nginx (frontend) is exposed to the public internet on ports 80/3000 (and 443 for HTTPS).
- **Network Isolation**: PostgreSQL and Redis have **no published host ports** (`expose` only) and can only be accessed internally by the backend over `vm_network`.
- **Health Checks & Startup Sequencing**: Backend waits for both PostgreSQL and Redis to be healthy (`service_healthy`) before booting, preventing database connection drops. Frontend waits for Backend to be healthy.

---

## 2. Target Deployment Specifications

- **Cloud Provider**: Amazon Web Services (AWS)
- **Region**: `ap-southeast-2` (Sydney)
- **AMI**: Ubuntu Server 24.04 LTS (Noble Numbat), 64-bit (x86_64)
- **Instance Type**: `t3.medium` or `t3a.medium` (2 vCPU, 4.0 GiB RAM)
  - *Rationale*: Python with ML dependencies (`scikit-learn`, `shap`, `xgboost`, `pandas`) plus FastAPI, Uvicorn, PostgreSQL, Redis, and Nginx runs comfortably within 1.5 – 2.2 GB of active RAM. 4 GB provides ample buffer for traffic spikes and memory-efficient SHAP tree computations.
- **Storage**: 25–30 GB General Purpose SSD (gp3) with 3,000 IOPS baseline.

---

## 3. Environment Variables Inventory

Create a `.env` file from `.env.example` in the project root:

```bash
cp .env.example .env
```

| Variable | Description | Production Example |
| :--- | :--- | :--- |
| `APP_ENV` | Application environment mode | `production` |
| `APP_DEBUG` | Debug logging and reload flag | `false` |
| `APP_SECRET_KEY` | Secret key for cookies and session signing | `openssl rand -hex 32` |
| `FRONTEND_PORT` | Host port mapped to frontend Nginx | `80` (or `3000`) |
| `POSTGRES_DB` | PostgreSQL database name | `vehicle_maintenance` |
| `POSTGRES_USER` | PostgreSQL superuser | `vm_admin` |
| `POSTGRES_PASSWORD` | PostgreSQL secure password | Strong random password |
| `PRIMARY_LLM_PROVIDER` | LLM router primary provider | `groq` |
| `PRIMARY_LLM_MODEL` | Primary model identifier | `openai/gpt-oss-120b` |
| `PRIMARY_LLM_API_KEY` | Primary provider API key | `gsk_...` |
| `PRIMARY_LLM_BASE_URL` | Primary provider OpenAI-compatible URL | `https://api.groq.com/openai/v1` |
| `SECONDARY_LLM_PROVIDER`| Secondary failover provider | `groq` |
| `SECONDARY_LLM_MODEL` | Secondary model identifier | `llama-3.3-70b-versatile` |
| `SECONDARY_LLM_API_KEY` | Secondary provider API key | `gsk_...` |
| `SECONDARY_LLM_BASE_URL`| Secondary provider base URL | `https://api.groq.com/openai/v1` |
| `ML_MODEL_PATH` | Path to trained Random Forest model | `models/maintenance_model.joblib` |
| `ML_PIPELINE_PATH` | Path to preprocessor pipeline | `models/preprocessing_pipeline.joblib` |
| `FRONTEND_URL` | Public URL of application | `http://<ec2-public-ip>` or `https://domain.com` |
| `GOOGLE_REDIRECT_URI` | Google OAuth callback URL | `http://<ec2-public-ip>/auth/callback` |
| `GOOGLE_CLIENT_ID` | Optional Google OAuth Client ID | `client-id.apps.googleusercontent.com` |
| `GOOGLE_CLIENT_SECRET` | Optional Google OAuth Secret | `GOCSPX-...` |

---

## 4. Local Docker Setup & Verification

### Step 1: Ensure Docker Engine is Running
On Windows or macOS, ensure Docker Desktop is open. On Linux, ensure the Docker daemon is active:
```bash
sudo systemctl status docker
```

### Step 2: Build and Start Containers
```bash
# Build images and start all 4 containers in the background
docker compose up -d --build
```

### Step 3: Verify Container Status & Health
```bash
docker compose ps
```
All services (`vm_frontend`, `vm_backend`, `vm_postgres`, `vm_redis`) should show status `Up (healthy)`.

### Step 4: Test Key Endpoints Locally
- **Frontend SPA**: Open `http://localhost:3000` (or your configured `FRONTEND_PORT`).
- **System Health API**:
  ```bash
  curl http://localhost:3000/api/health
  # Expected: {"status":"healthy","components":{"ml_model":"loaded","llm_router":"configured","redis":"connected"}}
  ```
- **ML Health Check**:
  ```bash
  curl http://localhost:3000/api/health/ml
  # Expected: {"is_loaded":true,"metadata":{...}}
  ```
- **Interactive Swagger Docs**: `http://localhost:3000/docs`

---

## 5. AWS EC2 Step-by-Step Deployment Guide

### Step 1: Launch EC2 Instance in Sydney (`ap-southeast-2`)
1. Open the [AWS EC2 Console](https://ap-southeast-2.console.aws.amazon.com/ec2).
2. Click **Launch instance**.
3. Name: `vehicle-maintenance-ai-prod`.
4. AMI: **Ubuntu Server 24.04 LTS (HVM), SSD Volume Type**.
5. Architecture: `64-bit (x86)`.
6. Instance type: `t3.medium` (2 vCPU, 4.0 GiB RAM).
7. Key pair: Select or create your SSH key pair (`.pem`).
8. Storage: Configure 30 GiB `gp3` storage.

### Step 2: Configure EC2 Security Group
Create a security group with the following rules:

| Type | Protocol | Port Range | Source | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **SSH** | TCP | 22 | `My IP` (or bastion CIDR) | Secure admin SSH access |
| **HTTP** | TCP | 80 | `0.0.0.0/0` | Public web traffic & ACME challenge |
| **HTTPS**| TCP | 443 | `0.0.0.0/0` | Secure SSL web traffic |

> [!WARNING]
> Do NOT open ports 5432 (Postgres), 6379 (Redis), or 7860 (Backend) to `0.0.0.0/0`. All internal communication occurs inside the Docker bridge network.

### Step 3: Assign an Elastic IP (Recommended)
1. Go to **EC2 > Elastic IPs > Allocate Elastic IP address**.
2. Associate the allocated Elastic IP with your `vehicle-maintenance-ai-prod` instance to prevent IP changes across restarts.

### Step 4: Connect to Your EC2 Instance
```bash
ssh -i /path/to/your-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

### Step 5: Install Docker Engine & Docker Compose on Ubuntu 24.04
Run the following script to install official Docker Engine:

```bash
# Update package list and install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine and Docker Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Allow 'ubuntu' user to run docker without sudo
sudo usermod -aG docker ubuntu
newgrp docker

# Verify installation
docker --version
docker compose version
```

### Step 6: Clone Repository and Setup Environment
```bash
# Clone project repository
git clone <YOUR_GIT_REPOSITORY_URL> /home/ubuntu/app
cd /home/ubuntu/app

# Create production environment configuration
cp .env.example .env
nano .env
```
Inside `.env`:
- Set `FRONTEND_PORT=80` (binds frontend to standard HTTP port 80).
- Set `POSTGRES_PASSWORD` to a strong unique secret.
- Set `PRIMARY_LLM_API_KEY` and `SECONDARY_LLM_API_KEY`.
- Set `FRONTEND_URL=http://<YOUR_EC2_PUBLIC_IP>` (or your custom domain).

### Step 7: Build and Launch Containers
```bash
docker compose up -d --build
```

Monitor container boot and health status:
```bash
docker compose ps
docker compose logs -f
```

---

## 6. Configuring HTTPS & Custom Domain (Let's Encrypt)

If you have a domain name pointing to your EC2 instance (e.g., `autocare.yourdomain.com`):

### Option A: Certbot with Host Nginx
1. Set `FRONTEND_PORT=3000` in `.env` so Docker runs on port 3000.
2. Install Nginx and Certbot on Ubuntu host:
   ```bash
   sudo apt-get install -y nginx certbot python3-certbot-nginx
   ```
3. Configure host Nginx `/etc/nginx/sites-available/app`:
   ```nginx
   server {
       server_name autocare.yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:3000;
           proxy_http_version 1.1;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;

           # SSE Streaming Support
           proxy_buffering off;
           proxy_cache off;
           proxy_set_header Connection '';
           chunked_transfer_encoding off;
       }
   }
   ```
4. Enable site and obtain SSL certificate:
   ```bash
   sudo ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/
   sudo rm /etc/nginx/sites-enabled/default
   sudo nginx -t && sudo systemctl reload nginx
   sudo certbot --nginx -d autocare.yourdomain.com
   ```
Certbot automatically sets up automatic certificate renewal via systemd timer.

---

## 7. Operational Management & Maintenance

### Inspecting Logs
```bash
# Follow all container logs
docker compose logs -f

# Follow backend logs specifically
docker compose logs -f backend

# Follow frontend access & error logs
docker compose logs -f frontend
```

### Stopping and Restarting Services
```bash
# Graceful restart of backend
docker compose restart backend

# Stop all services (preserves database volume)
docker compose down

# Stop and remove volumes (CAUTION: Deletes database!)
# docker compose down -v
```

### Safe Application Updates
To update application code without losing persistent database records:
```bash
cd /home/ubuntu/app
git pull origin main

# Rebuild and recreate containers
docker compose up -d --build
```
Docker Compose performs in-place recreation of changed containers while preserving existing `postgres_data` and `redis_data` volumes.

---

## 8. Database Persistence & Backup Procedures

PostgreSQL data is persisted in the Docker volume `postgres_data` (`/var/lib/postgresql/data`).

### Creating a Database Backup
```bash
# Generate timestamped backup SQL
BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S).sql"
docker compose exec -T postgres pg_dump -U postgres vehicle_maintenance > "/home/ubuntu/backups/${BACKUP_NAME}"
gzip "/home/ubuntu/backups/${BACKUP_NAME}"
```

### Restoring from a Backup
```bash
gunzip < "/home/ubuntu/backups/backup_xxxx.sql.gz" | docker compose exec -T postgres psql -U postgres -d vehicle_maintenance
```

### Automated Nightly Cron Backup
Add the following to `crontab -e`:
```bash
0 2 * * * mkdir -p /home/ubuntu/backups && docker compose -f /home/ubuntu/app/docker-compose.yml exec -T postgres pg_dump -U postgres vehicle_maintenance | gzip > /home/ubuntu/backups/db_$(date +\%F).sql.gz
```

---

## 9. Performance & Resource Monitoring

### Real-Time Container Metrics
```bash
docker stats
```
Monitors instantaneous CPU %, Memory usage / limit, and Network I/O for each container.

### System-Wide Utilization
```bash
# Check CPU & RAM
htop

# Check Disk Space
df -h /
```

### Typical Resource Profile on 2 vCPU / 4 GB RAM:
- `vm_backend`: 300–450 MB RAM (Random Forest + SHAP TreeExplainer in memory)
- `vm_frontend`: 15–30 MB RAM (Nginx alpine)
- `vm_postgres`: 40–80 MB RAM (PostgreSQL 16)
- `vm_redis`: 10–25 MB RAM (Redis 7)
- **Total Memory Consumption**: ~500–700 MB out of 4,000 MB (~15% baseline).
- **Available Free RAM**: ~3.2 GB buffer for LLM routing and HTTP request spikes.
