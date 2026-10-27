// The voices MiniMax's Chinese system voices offer, plus an escape hatch
// for a cloned voice. The list mirrors what the Debug page offers so a
// voice chosen once works across every tab -- and the desktop pet, which
// has no room for a picker, uses the same default.

export interface VoiceOption {
  id: string;
  label: string;
}

export const VOICES: VoiceOption[] = [
  { id: "male-qn-qingse", label: "青涩青年（男）" },
  { id: "male-qn-jingying", label: "精英青年（男）" },
  { id: "male-qn-badao", label: "霸道青年（男）" },
  { id: "female-shaonv", label: "少女（女）" },
  { id: "female-yujie", label: "御姐（女）" },
  { id: "female-chengshu", label: "成熟女性（女）" },
  { id: "female-tianmei", label: "甜美女性（女）" },
];

export const CUSTOM_VOICE = "__custom__";

/** The voice the desktop pet uses. It has no room for a picker, so it
 *  takes a fixed one. */
export const DEFAULT_VOICE_ID = "female-shaonv";
