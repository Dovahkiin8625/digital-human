import { useEffect, useState } from "react";
import { listDriveModels } from "../api";
import type { DriveModelInfo } from "../types";

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

function fetchDriveModels(): Promise<DriveModelInfo[]> {
  if (cached) return Promise.resolve(cached);
  if (!inflight) {
    inflight = listDriveModels()
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
