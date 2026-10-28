// The voices MiniMax's Chinese system voices offer, plus an escape hatch
// for a cloned voice. The list mirrors what the Debug page offers so a
// voice chosen once works across every tab.

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

/** The voice the console starts with. */
export const DEFAULT_VOICE_ID = "female-shaonv";
