import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App as AntApp, ConfigProvider, theme as antdTheme } from "antd";
import zhCN from "antd/locale/zh_CN";
import App from "./App";
import { ThemeProvider, useTheme } from "./theme";
import { configureRuntime } from "./live2d/cubism/core";
import { getConsoleToken } from "./api";
import "./styles.css";

// Every Live2D request (core, shaders, model files) is routed through here so
// it can carry the server root and the token. The console is served from the
// same origin as the API, so only the token has to be configured.
configureRuntime({ baseUrl: "", token: getConsoleToken() });

function Root() {
  const { mode } = useTheme();
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: mode === "dark" ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        token: { colorPrimary: "#2b5cab", borderRadius: 8 },
      }}
    >
      {/* antd's App provides the context that message/modal/notification need
          so they pick up the active theme instead of the static default. */}
      <AntApp>
        <BrowserRouter basename="/admin">
          <App />
        </BrowserRouter>
      </AntApp>
    </ConfigProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <Root />
    </ThemeProvider>
  </React.StrictMode>
);