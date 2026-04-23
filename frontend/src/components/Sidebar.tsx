import { useState, useRef, useEffect, useCallback } from 'react';

interface Conversation {
  id: string;
  title: string;
  messages: { role: 'user' | 'assistant'; text: string }[];
}

interface SidebarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  theme: 'dark' | 'light';
}

const MIN_WIDTH = 180;
const MAX_WIDTH = 480;
const DEFAULT_WIDTH = 256;

export function Sidebar({
  conversations,
  activeConversationId,
  onSelect,
  onNew,
  onRename,
  onDelete,
  theme,
}: SidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const [search, setSearch] = useState('');
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [width, setWidth] = useState(() => {
    const saved = localStorage.getItem('gr_sidebar_width');
    return saved ? parseInt(saved) : DEFAULT_WIDTH;
  });
  const [isDragging, setIsDragging] = useState(false);

  const menuRef = useRef<HTMLDivElement>(null);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(0);

  const dk = theme === 'dark';

  const sidebarBg   = dk ? 'bg-gray-800 text-gray-100' : 'bg-white text-slate-800 border-r border-slate-200';
  const searchBg    = dk ? 'bg-gray-700 text-gray-100 placeholder-gray-400 border-gray-600' : 'bg-slate-100 text-slate-800 placeholder-slate-400 border-slate-300';
  const convoBg     = dk ? 'hover:bg-gray-700' : 'hover:bg-slate-100';
  const convoActive = dk ? 'bg-gray-700' : 'bg-slate-200';
  const menuBg      = dk ? 'bg-gray-700 border-gray-600' : 'bg-white border-slate-200';
  const menuItem    = dk ? 'hover:bg-gray-600 text-gray-100' : 'hover:bg-slate-100 text-slate-800';
  const inputBg     = dk ? 'bg-gray-700 text-gray-100 border-gray-600' : 'bg-white text-slate-900 border-slate-300';
  const dotsBg      = dk ? 'hover:bg-gray-600 text-gray-400' : 'hover:bg-slate-200 text-slate-500';
  const dragHandle  = dk ? 'bg-gray-700 hover:bg-emerald-500' : 'bg-slate-200 hover:bg-emerald-400';

  // Close menu on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpenMenuId(null);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // ✅ Drag logic
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragStartX.current = e.clientX;
    dragStartWidth.current = width;
    setIsDragging(true);
  }, [width]);

  useEffect(() => {
    if (!isDragging) return;

    function onMouseMove(e: MouseEvent) {
      const delta = e.clientX - dragStartX.current;
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, dragStartWidth.current + delta));
      setWidth(newWidth);
    }

    function onMouseUp() {
      setIsDragging(false);
      localStorage.setItem('gr_sidebar_width', String(width));
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    return () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, [isDragging, width]);

  // Save width to localStorage when drag ends
  useEffect(() => {
    if (!isDragging) {
      localStorage.setItem('gr_sidebar_width', String(width));
    }
  }, [isDragging, width]);

  function saveEdit(id: string) {
    onRename(id, editText || 'Untitled');
    setEditingId(null);
  }

  const filtered = conversations.filter((c) =>
    c.title.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <aside
      className={`flex-shrink-0 h-screen flex relative ${sidebarBg}`}
      style={{ width: `${width}px` }}
    >
      {/* Main sidebar content */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Top — fixed */}
        <div className="flex-shrink-0 p-3 border-b border-gray-700/30">
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-sm">Conversations</span>
            <button
              onClick={onNew}
              className="text-xs px-2 py-1 bg-emerald-600 hover:bg-emerald-700 text-white rounded"
            >
              New
            </button>
          </div>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search conversations..."
            className={`w-full text-xs px-3 py-2 rounded border focus:outline-none focus:ring-1 focus:ring-emerald-500 ${searchBg}`}
          />
        </div>

        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1" ref={menuRef}>
          {filtered.length === 0 && (
            <div className="text-xs text-center mt-4 opacity-50">
              {search ? 'No results found' : 'No conversations yet'}
            </div>
          )}

          {filtered.map((c) => (
            <div
              key={c.id}
              onClick={() => { onSelect(c.id); setOpenMenuId(null); }}
              className={`group relative p-2 rounded cursor-pointer text-sm transition-colors ${
                c.id === activeConversationId ? convoActive : convoBg
              }`}
            >
              {editingId === c.id ? (
                <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
                  <input
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && saveEdit(c.id)}
                    className={`flex-1 text-xs px-2 py-1 rounded border ${inputBg}`}
                    autoFocus
                  />
                  <button onClick={() => saveEdit(c.id)} className="text-xs px-1.5 py-1 bg-emerald-600 text-white rounded">Save</button>
                  <button onClick={() => setEditingId(null)} className="text-xs px-1.5 py-1 bg-gray-600 text-white rounded">X</button>
                </div>
              ) : (
                <div className="flex items-center justify-between gap-1">
                  <span className="truncate flex-1 text-xs">{c.title}</span>

                  {/* 3-dot menu */}
                  <div className="relative flex-shrink-0" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => setOpenMenuId(openMenuId === c.id ? null : c.id)}
                      className={`p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity ${dotsBg}`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                        <circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/>
                      </svg>
                    </button>

                    {openMenuId === c.id && (
                      <div className={`absolute right-0 top-6 z-50 w-36 rounded-lg shadow-lg border overflow-hidden ${menuBg}`}>
                        <button
                          onClick={() => { setEditingId(c.id); setEditText(c.title); setOpenMenuId(null); }}
                          className={`w-full text-left text-xs px-3 py-2 flex items-center gap-2 ${menuItem}`}
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                          Rename
                        </button>
                        <button
                          onClick={() => { onDelete(c.id); setOpenMenuId(null); }}
                          className="w-full text-left text-xs px-3 py-2 flex items-center gap-2 text-red-400 hover:bg-red-600 hover:text-white"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
                          Delete
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ✅ Drag handle — right edge of sidebar */}
      <div
        onMouseDown={onMouseDown}
        className={`absolute right-0 top-0 h-full w-px cursor-col-resize transition-colors z-10 ${isDragging ? 'bg-emerald-500' : dk ? 'bg-gray-600 hover:bg-emerald-500' : 'bg-slate-200 hover:bg-emerald-400'}`}
        title="Drag to resize"
      />

      {/* ✅ Full-screen drag overlay — prevents text selection during drag */}
      {isDragging && (
        <div className="fixed inset-0 z-50 cursor-col-resize select-none" />
      )}
    </aside>
  );
}
