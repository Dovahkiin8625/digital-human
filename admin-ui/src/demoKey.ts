import { api } from "./api";

const KEY = "dh_demo_key";

interface TenantRow {
  tenant_id: string;
  name: string;
}

/**
 * Return a customer-facing API key for the debug console, cached in localStorage.
 *
 * It must be the key of the tenant named "default": avatars uploaded from this
 * console land in that tenant (POST /api/admin/avatars defaults to it) and
 * create_session requires the avatar to belong to the key's tenant. Issuing a
 * fresh key for that existing tenant (rather than creating a new tenant on every
 * cleared localStorage) keeps the two surfaces pointed at the same tenant.
 */
export async function ensureDemoKey(): Promise<string> {
  const cached = localStorage.getItem(KEY);
  if (cached) return cached;

  const tenants = await api<TenantRow[]>("/api/admin/tenants");
  let tenant = tenants.find((t) => t.name === "default");
  if (!tenant) {
    tenant = await api<TenantRow>("/api/admin/tenants", {
      method: "POST",
      body: { name: "default" },
    });
  }

  // Rotate only the "default" tenant; never any other tenant.
  const issued = await api<{ api_key: string }>(
    `/api/admin/tenants/${tenant.tenant_id}/api-key`,
    { method: "POST" }
  );
  localStorage.setItem(KEY, issued.api_key);
  return issued.api_key;
}
