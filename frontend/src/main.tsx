import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { logger } from "./services/logger";
import "./index.scss";

// Initialize frontend logging (captures console, errors, and sends to backend)
logger.init();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
