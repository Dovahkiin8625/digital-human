import { NavLink, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { getToken, setToken } from "./api";
import Login from "./pages/Login";
import Avatars from "./pages/Avatars";
import AvatarDetail from "./pages/AvatarDetail";
import Debug from "./pages/Debug";
import System from "./pages/System";

function RequireAuth({ children }: { children: JSX.Element }) {
  return getToken() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const navigate = useNavigate();
  return (
    <div className="app">
      <nav>
        <NavLink to="/avatars">形象管理</NavLink>
        <NavLink to="/debug">会话调试</NavLink>
        <NavLink to="/system">系统</NavLink>
        <button
          onClick={() => {
            setToken(null);
            navigate("/login");
          }}
        >
          退出
        </button>
      </nav>
      <main>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/avatars" element={<RequireAuth><Avatars /></RequireAuth>} />
          <Route path="/avatars/:id" element={<RequireAuth><AvatarDetail /></RequireAuth>} />
          <Route path="/debug" element={<RequireAuth><Debug /></RequireAuth>} />
          <Route path="/system" element={<RequireAuth><System /></RequireAuth>} />
          <Route path="*" element={<Navigate to="/avatars" replace />} />
        </Routes>
      </main>
    </div>
  );
}
