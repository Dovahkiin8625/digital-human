import { Empty, Progress, Space, Typography } from "antd";

// Histogram buckets from /api/admin/sessions metrics.latency_histogram.
// Keys are ms ranges (e.g. "<25", "25-50", "50-100", "100-250", ...).
// Each row shows one bucket with its own `<Progress>` bar; width follows the
// largest bucket so buckets are visually comparable within one render.

export default function LatencyHistogram({ data }: { data?: Record<string, number> }) {
  if (!data) return <Empty description="暂无延迟分布数据" />;
  const entries = Object.entries(data);
  if (entries.length === 0) return <Empty description="暂无延迟分布数据" />;
  const max = entries.reduce((acc, [, v]) => Math.max(acc, v), 1);

  return (
    <Space direction="vertical" size={8} style={{ width: "100%" }}>
      {entries.map(([bucket, count]) => {
        // Clicking a bucket is not wired: progress bars are read-only here, no
        // tooltips needed beyond the bucket label itself.
        const pct = Math.round((count / max) * 100);
        return (
          <div key={bucket} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <Typography.Text style={{ width: 64, textAlign: "right" }}>{bucket}ms</Typography.Text>
            <Progress
              percent={pct}
              size="small"
              style={{ flex: 1, margin: 0 }}
              format={() => `${count}`}
              strokeColor="#2b5cab"
            />
          </div>
        );
      })}
    </Space>
  );
}
