import type { PredictResponse } from "@/lib/api";
import { formatInterval, formatPace, formatTempF, formatTime } from "@/lib/format";

type PredictionsTableProps = {
  result: PredictResponse;
};

export function PredictionsTable({ result }: PredictionsTableProps) {
  return (
    <section className="space-y-4">
      <div className="text-sm text-zinc-400">
        {result.race_name ? (
          <span className="font-medium text-zinc-200">{result.race_name}</span>
        ) : (
          "Manual conditions"
        )}
        {" · "}
        As of {result.as_of} · {formatTempF(result.temp_f)}°F ({result.temp_source})
        {result.elev_source ? ` · Elev: ${result.elev_source}` : ""}
      </div>

      {result.warnings.length > 0 && (
        <ul className="space-y-1 rounded-lg border border-amber-900/60 bg-amber-950/30 px-4 py-3 text-sm text-amber-100">
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      <div className="overflow-x-auto rounded-xl border border-zinc-800">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-zinc-900/80 text-zinc-400">
            <tr>
              <th className="px-4 py-3 font-medium">Distance</th>
              <th className="px-4 py-3 font-medium">Predicted</th>
              <th className="px-4 py-3 font-medium">Pace</th>
              <th className="px-4 py-3 font-medium">80% Interval</th>
              <th className="px-4 py-3 font-medium">Conf.</th>
            </tr>
          </thead>
          <tbody>
            {result.predictions.map((row) => (
              <tr key={row.distance_label} className="border-t border-zinc-800">
                <td className="px-4 py-3">{row.distance_label}</td>
                <td className="px-4 py-3">{formatTime(row.predicted_time_sec)}</td>
                <td className="px-4 py-3">{formatPace(row.pace_min_per_mi)}</td>
                <td className="px-4 py-3">
                  {formatInterval(row.interval_low_sec, row.interval_high_sec)}
                </td>
                <td className="px-4 py-3">{row.confidence}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
