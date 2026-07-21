import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 600_000,
});

/* ────────────────────────── Types ────────────────────────── */

export interface ShapeInfo {
  rows: number;
  columns: number;
  memory_kb: number;
}

export interface DtypeInfo {
  column: string;
  raw_dtype: string;
  inferred_type: string;
}

export interface MissingnessInfo {
  column: string;
  missing_pct: number;
  flagged: boolean;
}

export interface CardinalityInfo {
  column: string;
  unique_values: number;
  unique_ratio: number;
  flagged_high_cardinality: boolean;
}

export interface CorrelatedPair {
  col_a: string;
  col_b: string;
  r: number;
}

export interface OutlierInfo {
  column: string;
  outlier_count: number;
  outlier_pct: number;
  lower_bound: number;
  upper_bound: number;
}

export interface DuplicateRowInfo {
  duplicate_row_count: number;
  duplicate_row_pct: number;
  duplicate_row_indices: number[];
}

export interface DuplicateColumnPair {
  col_a: string;
  col_b: string;
}

export interface DuplicateColumnInfo {
  duplicate_column_pairs: DuplicateColumnPair[];
}

export interface ConstantFeatureInfo {
  column: string;
  fully_constant: boolean;
  constant_excluding_missing: boolean;
  constant_value: string | number | boolean | null;
}

export interface InfiniteValueInfo {
  column: string;
  infinite_count: number;
}

export interface RareCategoryDetail {
  value: string;
  pct: number;
}

export interface RareCategoryInfo {
  column: string;
  rare_categories: RareCategoryDetail[];
  rare_category_count: number;
}

export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface NumericChartData {
  column: string;
  q1: number;
  median: number;
  q3: number;
  lower_bound: number;
  upper_bound: number;
  histogram: HistogramBin[];
}

export interface CategoricalChartData {
  column: string;
  value_counts: { value: string; count: number }[];
}

export interface ScatterSample {
  col_a: string;
  col_b: string;
  r: number;
  sample: { x: number; y: number }[];
}

export interface ChartData {
  numeric: NumericChartData[];
  categorical: CategoricalChartData[];
  correlation_matrix: { col_a: string; col_b: string; r: number }[];
  scatter_samples: ScatterSample[];
}

export interface Diagnosis {
  shape: ShapeInfo;
  dtypes: DtypeInfo[];
  missingness: MissingnessInfo[];
  cardinality: CardinalityInfo[];
  correlated_pairs: CorrelatedPair[];
  outliers: OutlierInfo[];
  duplicate_rows: DuplicateRowInfo;
  duplicate_columns: DuplicateColumnInfo;
  constant_features: ConstantFeatureInfo[];
  infinite_values: InfiniteValueInfo[];
  rare_categories: RareCategoryInfo[];
  chart_data?: ChartData;
}

export interface UploadResponse {
  session_id: string;
  diagnosis: Diagnosis;
}

export interface SemanticType {
  column: string;
  semantic_type: string;
  is_identifier: boolean;
  notes: string;
}

export interface Recommendation {
  column: string;
  issue: string;
  severity: "high" | "medium" | "low";
  recommended_action: string;
  justification: string;
  confidence: number;
  needs_review?: boolean;
}

export interface LeakageFlag {
  column: string;
  score: number;
  reason: string;
}

export interface ClassBalance {
  class_counts_pct: Record<string, number>;
  is_imbalanced: boolean;
  minority_class: string;
}

export interface FeatureRelevance {
  column: string;
  score: number;
  semantic_type: string;
}

export interface TargetAnalysis {
  problem_type: "classification" | "regression";
  target_column: string;
  leakage_flags: LeakageFlag[];
  class_balance: ClassBalance | null;
  feature_relevance: FeatureRelevance[];
}

export interface AnalyzeResponse {
  semantic_types: SemanticType[];
  recommendations: Recommendation[];
  target_analysis?: TargetAnalysis;
}

export interface ColumnStats {
  missing_count: number;
  missing_pct: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
}

export interface ApplyActionResponse {
  success: boolean;
  column: string;
  action: string;
  before: ColumnStats;
  after: ColumnStats;
  full_diagnosis: Diagnosis;
}

export interface ActionSummary {
  column: string;
  action: string;
  justification: string;
  before: ColumnStats;
  after: ColumnStats;
}

export interface SummaryResponse {
  original_shape: { rows: number; columns: number };
  final_shape: { rows: number; columns: number };
  actions_applied: ActionSummary[];
  export_ready: boolean;
}

/* ────────────────────────── API Calls ────────────────────────── */

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post<UploadResponse>("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function analyzeData(
  sessionId: string,
  targetColumn?: string,
  targetPurpose?: string
): Promise<AnalyzeResponse> {
  const payload: Record<string, string> = { session_id: sessionId };
  if (targetColumn) payload.target_column = targetColumn;
  if (targetPurpose) payload.target_purpose = targetPurpose;

  const { data } = await api.post<AnalyzeResponse>("/analyze", payload);
  return data;
}

export async function applyAction(
  sessionId: string,
  column: string,
  action: string,
  justification: string
): Promise<ApplyActionResponse> {
  const { data } = await api.post<ApplyActionResponse>("/apply-action", {
    session_id: sessionId,
    column,
    action,
    justification,
  });
  return data;
}

export async function getRecommendations(sessionId: string): Promise<Recommendation[]> {
  const res = await api.get(`/analyze/${sessionId}`);
  return res.data.recommendations;
}

export async function fetchSummary(sessionId: string): Promise<SummaryResponse> {
  const { data } = await api.get<SummaryResponse>(`/session/${sessionId}/summary`);
  return data;
}

export async function exportData(sessionId: string): Promise<void> {
  const url = getExportUrl(sessionId);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Export failed: ${response.statusText}`);
  }
  const blob = await response.blob();
  const blobUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = blobUrl;
  link.download = `datarx_cleaned_${sessionId}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(blobUrl);
}

export function getExportUrl(sessionId: string): string {
  return `${API_BASE}/export/${sessionId}`;
}
