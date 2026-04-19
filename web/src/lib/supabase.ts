import { createClient, SupabaseClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return url.startsWith('http://') || url.startsWith('https://');
  } catch {
    return false;
  }
}

export const isSupabaseConfigured = (): boolean => {
  return isValidUrl(supabaseUrl) && supabaseAnonKey.length > 10;
};

let _supabase: SupabaseClient | null = null;

export const supabase = (() => {
  if (_supabase) return _supabase;
  if (isSupabaseConfigured()) {
    _supabase = createClient(supabaseUrl, supabaseAnonKey);
    return _supabase;
  }
  return null;
})();
