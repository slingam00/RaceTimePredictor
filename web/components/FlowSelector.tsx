"use client";

export type UserFlow = "search" | "manual";

type FlowSelectorProps = {
  value: UserFlow;
  onChange: (flow: UserFlow) => void;
};

const FLOWS: { id: UserFlow; label: string; description: string }[] = [
  {
    id: "search",
    label: "Find my race",
    description: "Search RunSignup for an upcoming race and predict from course data.",
  },
  {
    id: "manual",
    label: "Enter conditions manually",
    description: "No race listing? Enter elevation, temperature, and date yourself.",
  },
];

export function FlowSelector({ value, onChange }: FlowSelectorProps) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {FLOWS.map((flow) => {
        const selected = value === flow.id;
        return (
          <button
            key={flow.id}
            type="button"
            onClick={() => onChange(flow.id)}
            className={`rounded-xl border px-4 py-4 text-left transition ${
              selected
                ? "border-emerald-600 bg-emerald-950/30 ring-1 ring-emerald-600/60"
                : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-900/80"
            }`}
          >
            <div className="font-medium text-zinc-100">{flow.label}</div>
            <p className="mt-1 text-sm text-zinc-400">{flow.description}</p>
          </button>
        );
      })}
    </div>
  );
}
