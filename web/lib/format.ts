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
