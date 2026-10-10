import { Navigate, Outlet, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { Button, Layout, Menu, Popconfirm, Space, Tooltip, theme as antdTheme } from "antd";
import type { MenuProps } from "antd";
import { LogoutOutlined, MoonOutlined, SunOutlined } from "@ant-design/icons";
import { getToken, setToken } from "./api";
import { useTheme } from "./theme";
import Login from "./pages/Login";
import Avatars from "./pages/Avatars";
import AvatarDetail from "./pages/AvatarDetail";
import Debug from "./pages/Debug";
import System from "./pages/System";

function RequireAuth({ children }: { children: JSX.Element }) {
  return getToken() ? children : <Navigate to="/login" replace />;
}

const NAV: MenuProps["items"] = [
  { key: "/avatars", label: "形象管理" },
  { key: "/debug", label: "会话调试" },
  { key: "/system", label: "系统" },
];

/** Chrome shared by every signed-in page: brand, nav, theme switch, sign-out. */
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
        <Space size={4}>
          <Tooltip title={mode === "dark" ? "切换到亮色" : "切换到暗色"}>
            <Button
              type="text"
              aria-label="切换主题"
              icon={mode === "dark" ? <SunOutlined /> : <MoonOutlined />}
              onClick={toggle}
            />
          </Tooltip>
          <Popconfirm
            title="退出登录？"
            okText="退出"
            cancelText="取消"
            onConfirm={() => {
              setToken(null);
              navigate("/login", { replace: true });
            }}
          >
            <Button type="text" icon={<LogoutOutlined />}>
              退出
            </Button>
          </Popconfirm>
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
      <Route path="/login" element={<Login />} />
      <Route
        element={
          <RequireAuth>
            <Shell />
          </RequireAuth>
        }
      >
        <Route path="/avatars" element={<Avatars />} />
        <Route path="/avatars/:id" element={<AvatarDetail />} />
        <Route path="/debug" element={<Debug />} />
        <Route path="/system" element={<System />} />
      </Route>
      <Route path="*" element={<Navigate to="/avatars" replace />} />
    </Routes>
  );
}
