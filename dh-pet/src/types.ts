/**
 * The pet's view of the server's API.
 *
 * These are the shapes the pet actually reads -- the rest of the service's
 * vocabulary belongs to the console. The pet talks to the server over HTTP
 * only; nothing here is shared code with the console, so a change to one
 * side's types does not ripple into the other.
 */

/** One avatar row, as the business plane lists them. */
export interface AvatarInfo {
  avatar_id: string;
  name: string;
  status: "pending" | "processing" | "ready" | "failed";
  /** The digital-human model that drives this avatar. */
  drive_model: string;
  /** Whether that model is rendered by the client. */
  client_rendered: boolean;
  /** The pipeline's pack.json. */
  config: {
    warnings?: string[];
    [key: string]: unknown;
  };
  error: string | null;
  created_at: string;
}

/**
 * A Live2D model's editable configuration.
 *
 * ``model.entry`` is the model3.json filename inside the model directory;
 * the client resolves every sibling file (moc3, textures, motions)
 * relative to it.
 */
export interface Live2DModelConfig {
  model: {
    entry: string;
    scale?: number;
    y?: number;
  };
  /** Emotion tag -> index into the model's OWN expression list. */
  expressions?: Record<string, number>;
  [key: string]: unknown;
}

/** Whether the Cubism SDK is staged. Never errors: a missing SDK is
 *  something to display, not to throw on. */
export interface Live2DHealth {
  enabled: boolean;
  sdk_dir: string;
  sdk_present: boolean;
  core_present: boolean;
  shaders_present: boolean;
}
