import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Alert, Button, Card, Form, Input, Typography } from "antd";
import { LockOutlined } from "@ant-design/icons";
import { login, setToken } from "../api";

export default function Login() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  async function onFinish({ password }: { password: string }) {
    setLoading(true);
    setError("");
    try {
      const { token } = await login(password);
      setToken(token);
      navigate("/avatars", { replace: true });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
      <Card style={{ width: "100%", maxWidth: 380 }}>
        <Typography.Title level={3} style={{ textAlign: "center", marginBottom: 4 }}>
          数字人管理后台
        </Typography.Title>
        <Typography.Paragraph type="secondary" style={{ textAlign: "center" }}>
          请输入管理员密码以继续
        </Typography.Paragraph>
        <Form layout="vertical" requiredMark={false} onFinish={onFinish}>
          <Form.Item name="password" rules={[{ required: true, message: "请输入管理员密码" }]}>
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Admin 密码"
              size="large"
              autoFocus
              autoComplete="current-password"
            />
          </Form.Item>
          {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 16 }} />}
          <Button type="primary" htmlType="submit" size="large" block loading={loading}>
            登录
          </Button>
        </Form>
      </Card>
    </div>
  );
}
