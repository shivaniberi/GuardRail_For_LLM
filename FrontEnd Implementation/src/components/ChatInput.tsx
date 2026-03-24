import { Send } from 'lucide-react';

interface ChatInputProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
}

export function ChatInput({ value, onChange, onSubmit, disabled }: ChatInputProps) {
  return (
    <div className="w-full">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Type your prompt here..."
        rows={4}
        className="w-full bg-gray-800 text-gray-100 p-3 rounded border border-gray-700 focus:outline-none focus:ring-2 focus:ring-emerald-500"
      />
      <div className="flex justify-end mt-2">
        <button
          onClick={onSubmit}
          disabled={disabled}
          className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white px-4 py-2 rounded"
        >
          <Send />
          Submit
        </button>
      </div>
    </div>
  );
}
