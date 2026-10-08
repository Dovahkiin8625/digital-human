import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login, setToken } from "../api";

export default function Login() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    try {
      const { token } = await login(password);
      setToken(token);
      navigate("/avatars");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <form className="card" onSubmit={onSubmit}>
      <h2>管理员登录</h2>
      <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Admin 密码" />
      <button type="submit">登录</button>
      {error && <p className="error">{error}</p>}
    </form>
  );
}
