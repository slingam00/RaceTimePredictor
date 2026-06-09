export function formatTime(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

export function formatPace(minPerMile: number): string {
  if (minPerMile <= 0 || minPerMile > 60) {
    return "—";
  }
  const minutes = Math.floor(minPerMile);
  let seconds = Math.round((minPerMile - minutes) * 60);
  let adjMinutes = minutes;
  if (seconds === 60) {
    adjMinutes += 1;
    seconds = 0;
  }
  return `${adjMinutes}:${String(seconds).padStart(2, "0")}/mi`;
}

export function formatInterval(lowSec: number, highSec: number): string {
  return `${formatTime(lowSec)} – ${formatTime(highSec)}`;
}

export function formatRaceDate(value: string | null | undefined): string {
  if (!value) return "Date TBD";
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [year, month, day] = value.split("-");
    return `${month}/${day}/${year}`;
  }
  return value;
}

export function formatTempF(tempF: number): string {
  const rounded = Math.round(tempF * 10) / 10;
  return Number.isInteger(rounded) ? `${rounded}` : rounded.toFixed(1);
}

export function formatLocation(city?: string | null, state?: string | null): string {
  if (city && state) return `${city}, ${state}`;
  return city || state || "Location TBD";
}
