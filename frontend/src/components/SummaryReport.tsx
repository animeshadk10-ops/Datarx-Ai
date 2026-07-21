"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { fetchSummary, SummaryResponse, Recommendation } from "@/lib/api";
import ExportButton from "./ExportButton";
import ExportSuccessModal from "./ExportSuccessModal";

interface SummaryReportProps {
  sessionId: string;
  skippedIssues: Recommendation[];
  onReset: () => void;
}

export default function SummaryReport({ sessionId, skippedIssues, onReset }: SummaryReportProps) {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    async function loadSummary() {
      try {
        const data = await fetchSummary(sessionId);
        setSummary(data);
      } catch (err) {
        setError("Failed to load summary. Please try again.");
      } finally {
        setLoading(false);
      }
    }
    loadSummary();
  }, [sessionId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4">
        <div className="w-8 h-8 border-4 border-[var(--accent-primary)] border-t-transparent rounded-full animate-spin" />
        <p className="text-text-muted font-medium">Generating final report...</p>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="p-6 bg-[var(--danger)]/10 border border-[var(--danger)]/30 rounded-xl text-[var(--danger)] text-center font-medium">
        {error || "Could not load summary."}
      </div>
    );
  }

  // Helper to format stat changes
  const formatStat = (before: number, after: number) => {
    if (before === after) return <span className="text-text-secondary">{before}</span>;
    return (
      <span className="flex items-center gap-3">
        <span className="text-text-secondary line-through opacity-60 text-xl">{before}</span>
        <span className="text-[var(--accent-primary)] font-bold">{after}</span>
      </span>
    );
  };

  // Filter skipped issues: remove those that were applied
  const actualSkipped = skippedIssues.filter(issue => 
    !summary.actions_applied.some(
      action => action.column === issue.column && action.action === issue.recommended_action
    )
  );

  return (
    <div className="max-w-4xl mx-auto space-y-10 animate-fade-in">
      <div className="text-center space-y-3">
        <h2 className="text-3xl font-extrabold text-text-primary">Cleaning Summary</h2>
        <p className="text-text-muted max-w-xl mx-auto">
          Here is a complete record of the changes made to your dataset during this session.
        </p>
      </div>

      {/* Dataset Shape Changes */}
      <div className="glass-card p-6 md:p-8 flex flex-col md:flex-row items-center justify-around gap-6 text-center">
        <div>
          <p className="text-sm text-text-muted uppercase tracking-wider mb-2 font-semibold">Rows</p>
          <div className="text-3xl">
            {formatStat(summary.original_shape.rows, summary.final_shape.rows)}
          </div>
        </div>
        <div className="w-px h-12 bg-border-subtle hidden md:block" />
        <div>
          <p className="text-sm text-text-muted uppercase tracking-wider mb-2 font-semibold">Columns</p>
          <div className="text-3xl">
            {formatStat(summary.original_shape.columns, summary.final_shape.columns)}
          </div>
        </div>
      </div>

      {/* Actions Applied */}
      <div>
        <h3 className="text-xl font-bold text-text-primary mb-6 flex items-center gap-2">
          <span>✨</span> Actions Applied
        </h3>
        {summary.actions_applied.length === 0 ? (
          <div className="p-8 rounded-2xl bg-surface-elevated border border-border-subtle text-center shadow-sm">
            <p className="text-text-secondary text-lg">
              No changes were made — exporting your original data as-is.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {summary.actions_applied.map((action, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="p-6 rounded-2xl bg-surface-elevated border border-border-subtle shadow-sm flex flex-col md:flex-row gap-6 items-start"
              >
                <div className="flex-1 space-y-3">
                  <div className="inline-flex px-3 py-1 bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] border border-[var(--accent-primary)]/20 rounded-full text-xs font-bold uppercase tracking-wider">
                    {action.column}
                  </div>
                  <div>
                    <p className="text-lg font-bold text-text-primary">{action.action}</p>
                    <p className="text-sm text-text-secondary mt-1 leading-relaxed">{action.justification}</p>
                  </div>
                </div>
                
                {/* Before/After stat if available */}
                {(action.before.missing_count !== action.after.missing_count) && (
                  <div className="md:w-48 p-4 rounded-xl bg-surface border border-border-subtle shrink-0">
                    <p className="text-[10px] text-text-muted uppercase tracking-wider mb-2 font-bold">Missing Values</p>
                    <div className="flex items-center gap-3">
                      <span className="text-text-secondary line-through opacity-70">{action.before.missing_count}</span>
                      <span className="text-[var(--success)] font-extrabold text-lg">{action.after.missing_count}</span>
                    </div>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Skipped Issues */}
      {actualSkipped.length > 0 && (
        <div className="pt-4">
          <h3 className="text-xl font-bold text-text-primary mb-6 flex items-center gap-2 opacity-80">
            <span>⚠️</span> Issues Not Addressed
          </h3>
          <div className="space-y-3 opacity-90">
            {actualSkipped.map((issue, idx) => (
              <div key={idx} className="p-4 rounded-xl bg-surface border border-border-subtle flex items-start gap-4">
                <span className="text-xl mt-0.5">
                  {issue.severity === "high" ? "🚨" : issue.severity === "medium" ? "⚠️" : "ℹ️"}
                </span>
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-bold text-text-primary">{issue.column}</span>
                    <span className="text-text-muted text-sm font-medium">— {issue.issue}</span>
                  </div>
                  <p className="text-xs text-text-secondary font-medium">Skipped action: <span className="text-[var(--accent-primary)] opacity-80">{issue.recommended_action}</span></p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Final Export */}
      <div className="pt-10 border-t border-border-subtle flex justify-center pb-12">
        <ExportButton 
          sessionId={sessionId} 
          onSuccess={() => setIsModalOpen(true)}
        />
      </div>

      <ExportSuccessModal 
        isOpen={isModalOpen} 
        onStartNewSession={onReset} 
      />
    </div>
  );
}
