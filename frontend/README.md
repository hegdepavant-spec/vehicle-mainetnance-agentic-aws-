# AutoCare AI — Frontend Application

Modern, conversational React interface for the **Vehicle Maintenance Multi-Agent System**.

---

## 🚀 Features

- **ChatGPT-Style Conversational Interface**: Turn-by-turn chat with real-time Server-Sent Events (SSE) streaming.
- **Dynamic Assessment Cards**: Visual display of the 19-feature vehicle assessment state, known factors, and remaining questions.
- **ML Maintenance Prediction Cards**: Displays the Random Forest risk score (Low / High risk), class probabilities, and confidence notes.
- **SHAP Factor Explanations**: Visual contribution bars explaining why the model predicted maintenance required or not required.
- **Emergency Alert Modal**: Automatic popup when critical safety hazards are detected (`Brake_Condition = 'Worn Out'` or `Battery_Status = 'Weak'`), enabling 1-click Gmail emergency dispatch to emergency contacts.
- **Google OAuth 2.0 PKCE Integration**: Secure Google Sign-In with emergency contact management.

---

## 🛠️ Tech Stack

- **Framework**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS + Lucide Icons
- **Markdown & Streaming**: `react-markdown` with SSE stream handling
- **State Management**: React Hooks & Context API

---

## 💻 Local Development

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment
Create a `.env` file or rely on Vite defaults:
```env
VITE_API_BASE_URL=http://localhost:7860
```

### 3. Run Development Server
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📦 Build for Production

```bash
npm run build
```
The compiled production bundle is generated in `dist/` and served via Nginx in Docker deployments.
