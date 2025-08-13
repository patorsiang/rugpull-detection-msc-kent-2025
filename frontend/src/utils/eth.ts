export function parseAddresses(raw: string): string[] {
  if (!raw) return [];
  const parts = raw
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const isHexAddr = (s: string) => /^0x[a-fA-F0-9]{40}$/.test(s);
  return Array.from(new Set(parts.filter(isHexAddr)));
}
