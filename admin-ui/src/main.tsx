import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, HashRouter } from "react-router-dom";
import { App as AntApp, ConfigProvider, theme as antdTheme } from "antd";
import zhCN from "antd/locale/zh_CN";
import App from "./App";
import { ThemeProvider, useTheme } from "./theme";
import { configureRuntime } from "./live2d/cubism/core";
import { getConsoleToken } from "./api";
import "./styles.css";

// Electron loads this bundle from file://, where BrowserRouter has no server
// to resolve paths against. The preload publishes window.petConfig before any
// page script runs, so its presence is a reliable "this is the pet" signal.
const isPet = typeof window !== "undefined" && Boolean(window.petConfig);

// Every Live2D request (core, shaders, model files) is routed through here so
// it can carry the server root and the token. In the console those are
// same-origin and optional; the pet sets both before mounting a stage.
configureRuntime({ baseUrl: isPet ? undefined : "", token: getConsoleToken() });

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
        {isPet ? (
          <HashRouter>
            <App />
          </HashRouter>
        ) : (
          <BrowserRouter basename="/admin">
            <App />
          </BrowserRouter>
        )}
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