import { useEffect, useState } from "react";
import { Navigate, Outlet, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { Button, Layout, Menu, Space, Tooltip, theme as antdTheme } from "antd";
import type { MenuProps } from "antd";
import { MoonOutlined, SunOutlined } from "@ant-design/icons";
import { api } from "./api";
import { useTheme } from "./theme";
import ActivatePortrait from "./pages/ActivatePortrait";
import Avatars from "./pages/Avatars";
import AvatarDetail from "./pages/AvatarDetail";
import Debug from "./pages/Debug";
import System from "./pages/System";
import type { SystemInfo } from "./types";

const NAV: MenuProps["items"] = [
  { key: "/activate-portrait", label: "图生视频" },
  { key: "/avatars", label: "形象管理" },
  { key: "/debug", label: "会话调试" },
  { key: "/system", label: "系统" },
];

// 5-second poll keeps the header chip fresh enough to spot a "weights_present
// flipped" change without hammering /api/admin/system. The value is cached
// only to avoid a blank tooltip on first render; the displayed text always
// reflects whatever the latest successful poll returned.
function EngineStatusChip() {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  useEffect(() => {
    let alive = true;
    async function tick() {
      try {
        const data = await api<SystemInfo>("/api/admin/system");
        if (alive) setInfo(data);
      } catch {
        /* keep showing the last-known value on transient failures */
      }
    }
    tick();
    const t = window.setInterval(tick, 5000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, []);
  if (!info) {
    return <span style={{ fontSize: 12, color: "#999" }}>引擎…</span>;
  }
  const gpu = info.engine.gpu ? "GPU" : "CPU";
  const dir = info.data_dir || "";
  const version = info.version || "";
  // One-line tooltip: engine name, execution device, data dir, build version.
  // Each piece is a separate line so multi-segment paths stay readable.
  const title = (
    <div style={{ lineHeight: 1.6 }}>
      <div>引擎：{info.engine.name}</div>
      <div>设备：{gpu}</div>
      <div>数据目录：{dir}</div>
      <div>版本：{version}</div>
    </div>
  );
  return (
    <Tooltip title={title} placement="bottomRight">
      <span
        style={{
          fontSize: 12,
          padding: "2px 10px",
          borderRadius: 12,
          background: info.engine.weights_present === false ? "#faad1455" : "#52c41a33",
          color: info.engine.weights_present === false ? "#faad14" : "#52c41a",
          cursor: "default",
          whiteSpace: "nowrap",
        }}
      >
        {info.engine.name} · {gpu}
      </span>
    </Tooltip>
  );
}

/** Chrome shared by every signed-in page: brand, nav, theme switch. */
function Shell() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { mode, toggle } = useTheme();
  const { token } = antdTheme.useToken();

  const selected = NAV!.find((item) => item && pathname.startsWith(String(item.key)))?.key;

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Layout.Header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 24,
          paddingInline: 24,
          background: token.colorBgContainer,
          borderBottom: `1px solid ${token.colorBorderSecondary}`,
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 600, whiteSpace: "nowrap" }}>数字人管理后台</div>
        <Menu
          mode="horizontal"
          selectedKeys={selected ? [String(selected)] : []}
          items={NAV}
          onClick={({ key }) => navigate(key)}
          style={{ flex: 1, minWidth: 0, background: "transparent", borderBottom: "none" }}
        />
        <Space size={12}>
          <EngineStatusChip />
          <Tooltip title={mode === "dark" ? "切换到亮色" : "切换到暗色"}>
            <Button
              type="text"
              aria-label="切换主题"
              icon={mode === "dark" ? <SunOutlined /> : <MoonOutlined />}
              onClick={toggle}
            />
          </Tooltip>
        </Space>
      </Layout.Header>
      <Layout.Content style={{ padding: 24 }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <Outlet />
        </div>
      </Layout.Content>
    </Layout>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/activate-portrait" element={<ActivatePortrait />} />
        <Route path="/avatars" element={<Avatars />} />
        <Route path="/avatars/:id" element={<AvatarDetail />} />
        <Route path="/debug" element={<Debug />} />
        <Route path="/system" element={<System />} />
      </Route>
      <Route path="*" element={<Navigate to="/avatars" replace />} />
    </Routes>
  );
}