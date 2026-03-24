/*
  # Create Guardrail Evaluations and Logs Tables

  1. New Tables
    - `evaluations`
      - `id` (uuid, primary key)
      - `input_prompt` (text) - User's input text
      - `output_response` (text) - AI's response
      - `evaluation_results` (jsonb) - Safety check results
      - `overall_status` (text) - 'safe', 'warning', 'blocked'
      - `confidence_score` (numeric) - 0-100
      - `created_at` (timestamp)

    - `safety_checks`
      - `id` (uuid, primary key)
      - `evaluation_id` (uuid, foreign key)
      - `check_type` (text) - 'toxicity', 'hallucination', 'policy_violation', 'bias', 'pii'
      - `score` (numeric) - 0-100 confidence
      - `status` (text) - 'pass', 'warn', 'fail'
      - `details` (jsonb) - Additional metadata
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on both tables
    - Public access for demo

  3. Indexes
    - Index on evaluation_id for fast lookups
    - Index on created_at for sorting
    - Index on overall_status for filtering
*/

CREATE TABLE IF NOT EXISTS evaluations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  input_prompt text NOT NULL,
  output_response text,
  evaluation_results jsonb DEFAULT '{}'::jsonb,
  overall_status text NOT NULL DEFAULT 'processing',
  confidence_score numeric DEFAULT 0,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS safety_checks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  evaluation_id uuid NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
  check_type text NOT NULL,
  score numeric NOT NULL DEFAULT 0,
  status text NOT NULL,
  details jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE safety_checks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public access to evaluations"
  ON evaluations FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public insert to evaluations"
  ON evaluations FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Allow public update to evaluations"
  ON evaluations FOR UPDATE
  TO public
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Allow public access to safety_checks"
  ON safety_checks FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public insert to safety_checks"
  ON safety_checks FOR INSERT
  TO public
  WITH CHECK (true);

CREATE INDEX IF NOT EXISTS idx_safety_checks_evaluation_id ON safety_checks(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_overall_status ON evaluations(overall_status);
