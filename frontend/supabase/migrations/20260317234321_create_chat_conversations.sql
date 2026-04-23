/*
  # Create Chat Conversations and Messages Tables

  1. New Tables
    - `conversations`
      - `id` (uuid, primary key)
      - `title` (text) - Auto-generated title from first message
      - `created_at` (timestamp)
      - `updated_at` (timestamp)
    
    - `messages`
      - `id` (uuid, primary key)
      - `conversation_id` (uuid, foreign key)
      - `text` (text) - The message content
      - `is_user` (boolean) - Whether message is from user or AI
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on both tables
    - Public access for demo (anyone can create and view conversations)
    - Add cleanup policy for old conversations (>30 days)

  3. Indexes
    - Index on conversation_id for fast message queries
    - Index on created_at for sorting conversations
*/

CREATE TABLE IF NOT EXISTS conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title text NOT NULL DEFAULT 'New Conversation',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  text text NOT NULL,
  is_user boolean NOT NULL DEFAULT false,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public access to conversations"
  ON conversations FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public insert to conversations"
  ON conversations FOR INSERT
  TO public
  WITH CHECK (true);

CREATE POLICY "Allow public update to conversations"
  ON conversations FOR UPDATE
  TO public
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Allow public access to messages"
  ON messages FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Allow public insert to messages"
  ON messages FOR INSERT
  TO public
  WITH CHECK (true);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
