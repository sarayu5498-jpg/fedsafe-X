import React, { useState } from 'react';
import LoginOTP from './components/LoginOTP';
import ChatInterface from './components/ChatInterface';
import AdminDashboard from './components/AdminDashboard';

export default function App() {
    const [user1, setUser1] = useState(null);
    const [user2, setUser2] = useState(null);
    const [showAdminDashboard, setShowAdminDashboard] = useState(false);

    return (
        <div style={{
            backgroundColor: '#0f172a',
            minHeight: '100vh',
            padding: '1.25rem',
            fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
            color: '#f8fafc',
            boxSizing: 'border-box'
        }}>
            {/* Global Scrollbar styling */}
            <style>{`
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
      `}</style>

            {/* Header Bar */}
            <header style={{
                background: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '12px',
                padding: '1.25rem 1.5rem',
                marginBottom: '1.5rem',
                boxShadow: '0 4px 20px -2px rgba(0, 0, 0, 0.25)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <div>
                    <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '700', color: '#f8fafc', letterSpacing: '-0.3px' }}>
                        FedSafe-X Hub🌐
                    </h1>
                </div>

                <button
                    onClick={() => setShowAdminDashboard(!showAdminDashboard)}
                    style={{
                        backgroundColor: showAdminDashboard ? '#334155' : '#10b981',
                        color: showAdminDashboard ? '#f8fafc' : '#022c22',
                        border: 'none',
                        borderRadius: '8px',
                        padding: '0.6rem 1.1rem',
                        cursor: 'pointer',
                        fontWeight: '700',
                        fontSize: '0.85rem',
                        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
                        transition: 'all 0.2s ease'
                    }}>
                    📊 {showAdminDashboard ? "Return to Live Terminals" : "Admin Threat Analytics"}
                </button>
            </header>

            {/* Main Container View */}
            {showAdminDashboard ? (
                <AdminDashboard onClose={() => setShowAdminDashboard(false)} />
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', maxWidth: '1500px', margin: '0 auto' }}>
                    {/* Client Window A */}
                    <div style={{ backgroundColor: '#24344eff', border: '1px solid #334155', borderRadius: '12px', padding: '1.25rem', boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid #334155', paddingBottom: '0.6rem' }}>
                            <h2 style={{ color: '#38bdf8', fontSize: '1rem', margin: 0, fontWeight: '700' }}>💻 USER A</h2>
                        </div>
                        {!user1 ? <LoginOTP onLoginSuccess={(p) => setUser1(p)} windowName="User A" /> : <ChatInterface currentUser={user1} onBackToHome={() => setUser1(null)} />}
                    </div>

                    {/* Client Window B */}
                    <div style={{ backgroundColor: '#24344eff', border: '1px solid #334155', borderRadius: '12px', padding: '1.25rem', boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid #334155', paddingBottom: '0.6rem' }}>
                            <h2 style={{ color: '#a78bfa', fontSize: '1rem', margin: 0, fontWeight: '700' }}>💻 USER B</h2>
                        </div>
                        {!user2 ? <LoginOTP onLoginSuccess={(p) => setUser2(p)} windowName="User B" /> : <ChatInterface currentUser={user2} onBackToHome={() => setUser2(null)} />}
                    </div>
                </div>
            )}
        </div>
    );
}