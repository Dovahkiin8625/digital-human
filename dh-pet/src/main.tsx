import React from "react";
import ReactDOM from "react-dom/client";
import { App as AntApp, ConfigProvider, theme as antdTheme } from "antd";
import zhCN from "antd/locale/zh_CN";
import Pet from "./Pet";
import "./styles.css";

/**
 * The pet's entry point: one page, no router.
 *
 * The window is transparent and always-on-top, so there is no console
 * chrome here and no theme provider -- the console's paints the page
 * background from the active theme token, which on a transparent window is
 * a grey slab sitting on the desktop.
 *
 * The palette is fixed to the dark algorithm: the pet floats over whatever
 * the desktop shows, and light text with the shadow in styles.css stays
 * legible on both a light wallpaper and a dark one.
 */
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: antdTheme.darkAlgorithm,
        token: { colorPrimary: "#2b5cab", borderRadius: 8 },
      }}
    >
      {/* antd's App provides the context that message/modal/notification need
          so they pick up the active theme instead of the static default. */}
      <AntApp>
        <Pet />
      </AntApp>
    </ConfigProvider>
  </React.StrictMode>
);
