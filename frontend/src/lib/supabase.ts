import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseKey);

export interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
}

export async function createConversation(title: string): Promise<string> {
  const { data, error } = await supabase
    .from('conversations')
    .insert({ title })
    .select('id')
    .single();

  if (error) throw error;
  return data.id;
}

export async function saveMessage(
  conversationId: string,
  text: string,
  isUser: boolean
): Promise<void> {
  const { error } = await supabase
    .from('messages')
    .insert({
      conversation_id: conversationId,
      text,
      is_user: isUser,
    });

  if (error) throw error;

  const { error: updateError } = await supabase
    .from('conversations')
    .update({ updated_at: new Date().toISOString() })
    .eq('id', conversationId);

  if (updateError) throw updateError;
}

export async function getMessages(conversationId: string): Promise<Message[]> {
  const { data, error } = await supabase
    .from('messages')
    .select('*')
    .eq('conversation_id', conversationId)
    .order('created_at', { ascending: true });

  if (error) throw error;

  return data.map((msg: any) => ({
    id: msg.id,
    text: msg.text,
    isUser: msg.is_user,
    timestamp: new Date(msg.created_at),
  }));
}

export async function getConversations(): Promise<Conversation[]> {
  const { data, error } = await supabase
    .from('conversations')
    .select('*')
    .order('updated_at', { ascending: false })
    .limit(50);

  if (error) throw error;

  return data.map((conv: any) => ({
    id: conv.id,
    title: conv.title,
    createdAt: new Date(conv.created_at),
    updatedAt: new Date(conv.updated_at),
  }));
}

export async function updateConversationTitle(
  conversationId: string,
  title: string
): Promise<void> {
  const { error } = await supabase
    .from('conversations')
    .update({ title, updated_at: new Date().toISOString() })
    .eq('id', conversationId);

  if (error) throw error;
}
