import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from "react";
import { theme as antdTheme } from "antd";

export type ThemeMode = "dark" | "light";

const STORAGE_KEY = "dh-admin-theme";

function initialMode(): ThemeMode {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === "dark" || saved === "light") return saved;
  // First visit: follow the OS, falling back to the app's original dark look.
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

interface ThemeCtx {
  mode: ThemeMode;
  toggle: () => void;
}

const Ctx = createContext<ThemeCtx>({ mode: "dark", toggle: () => undefined });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>(initialMode);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, mode);
    // antd v5 is CSS-in-JS and only styles its own components; the page canvas
    // behind them (and overscroll areas) is ours to paint. Derive the colour
    // from the same token antd uses so the two never drift apart.
    const token = antdTheme.getDesignToken({
      algorithm: mode === "dark" ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
    });
    document.documentElement.style.colorScheme = mode;
    document.body.style.background = token.colorBgLayout;
  }, [mode]);

  const value = useMemo<ThemeCtx>(
    () => ({ mode, toggle: () => setMode((m) => (m === "dark" ? "light" : "dark")) }),
    [mode]
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTheme() {
  return useContext(Ctx);
}
