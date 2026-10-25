import { useEffect, useState } from "react";
import { api } from "../api";
import type { DriveModelId, DriveModelInfo } from "../types";

/**
 * The registered digital-human models, fetched once and shared.
 *
 * The registry only changes when the server restarts, so one fetch per
 * console session is enough. The cache also means the pages that need the
 * list -- the create form, the session pickers, the detail header -- all
 * read the same answer.
 */
let cached: DriveModelInfo[] | null = null;
let inflight: Promise<DriveModelInfo[]> | null = null;

export function fetchDriveModels(): Promise<DriveModelInfo[]> {
  if (cached) return Promise.resolve(cached);
  if (!inflight) {
    inflight = api<DriveModelInfo[]>("/api/admin/drive-models")
      .then((rows) => {
        cached = rows;
        return rows;
      })
      .finally(() => {
        inflight = null;
      });
  }
  return inflight;
}

/** The drive models, as a React value. Empty until the fetch lands. */
export function useDriveModels(): DriveModelInfo[] {
  const [models, setModels] = useState<DriveModelInfo[]>(cached ?? []);
  useEffect(() => {
    let alive = true;
    fetchDriveModels()
      .then((rows) => {
        if (alive) setModels(rows);
      })
      .catch(() => {
        /* the pages render what they have; the ids are still readable */
      });
    return () => {
      alive = false;
    };
  }, []);
  return models;
}

/** The operator-facing label of a drive model, or the raw id as a fallback. */
export function driveModelLabel(models: DriveModelInfo[], id: DriveModelId): string {
  return models.find((m) => m.name === id)?.label_zh ?? id;
}

/**
 * Whether a drive model is rendered by the client.
 *
 * Unknown ids answer false: the row then behaves like a server-driven
 * one, and the server refuses the session with a clear message rather
 * than the console guessing wrong.
 */
export function isClientRendered(models: DriveModelInfo[], id: DriveModelId): boolean {
  return models.find((m) => m.name === id)?.client_rendered ?? false;
}
