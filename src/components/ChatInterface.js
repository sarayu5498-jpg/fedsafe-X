import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_BASE = "http://localhost:8000/api";

export default function ChatInterface({ currentUser, onBackToHome }) {
    const [connectionStatus, setConnectionStatus] = useState('CHECKING');
    const [peerUser, setPeerUser] = useState(null);
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const [isBlocked, setIsBlocked] = useState(false);
    const [muteSeconds, setMuteSeconds] = useState(0);
    const [notification, setNotification] = useState(null);
    const chatScrollRef = useRef(null);

    const showNotice = (msg, type = 'error') => {
        setNotification({ msg, type });
    };

    useEffect(() => {
        if (chatScrollRef.current) {
            chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
        }
    }, [messages]);

    // Fast Polling Cycle
    useEffect(() => {
        const pollSession = async () => {
            try {
                const statusRes = await axios.get(`${API_BASE}/user/status?email_id=${currentUser.email_id}&mobile_number=${currentUser.mobile_number}`);
                if (statusRes.data.blocked) {
                    setIsBlocked(true);
                    return;
                }

                setMuteSeconds(statusRes.data.mute_remaining_seconds || 0);

                const connRes = await axios.post(`${API_BASE}/chat/check_connection`, {
                    my_email: currentUser.email_id,
                    friend_email: currentUser.friend_email
                });

                if (connRes.data.connected) {
                    setConnectionStatus('CONNECTED');
                    setPeerUser({
                        username: connRes.data.peer_username,
                        email_id: connRes.data.peer_email,
                        mobile_number: connRes.data.peer_mobile
                    });

                    const msgRes = await axios.get(`${API_BASE}/chat/messages?user1_email=${currentUser.email_id}&user2_email=${connRes.data.peer_email}`);
                    setMessages(prev => {
                        const serverMsgs = msgRes.data.messages;
                        if (serverMsgs.length >= prev.length) {
                            return serverMsgs;
                        }
                        return prev;
                    });
                } else {
                    setConnectionStatus('FAILED');
                }
            } catch (err) {
                console.error(err);
            }
        };

        pollSession();
        const interval = setInterval(pollSession, 1200);
        return () => clearInterval(interval);
    }, [currentUser]);

    const sendMessage = async () => {
        if (!inputText.trim() || isBlocked || muteSeconds > 0 || connectionStatus !== 'CONNECTED') return;

        const currentText = inputText;
        setInputText('');
        setNotification(null);

        const tempMsg = {
            id: Date.now(),
            sender_email: currentUser.email_id,
            sender_username: currentUser.username,
            receiver_email: peerUser.email_id,
            sender_mobile: currentUser.mobile_number,
            text: currentText,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            is_harassment: false
        };

        setMessages(prev => [...prev, tempMsg]);

        try {
            await axios.post(`${API_BASE}/chat/send_message`, {
                sender_email: currentUser.email_id,
                receiver_email: peerUser.email_id,
                sender_mobile: currentUser.mobile_number,
                message_text: currentText
            });
        } catch (err) {
            if (err.response?.status === 403) {
                setIsBlocked(true);
            } else if (err.response?.status === 429) {
                showNotice(err.response.data.detail);
            } else {
                showNotice("Delivery failed: " + (err.response?.data?.detail || err.message));
            }
        }
    };

    if (isBlocked) {
        return (
            <div style={{ backgroundColor: '#450a0a', border: '1px solid #ef4444', borderRadius: '10px', padding: '1.5rem', textAlign: 'center', color: '#fca5a5' }}>
                <h2 style={{ color: '#ef4444', marginTop: 0, fontSize: '1.1rem' }}>🚫 ACCOUNT PERMANENTLY BLOCKED</h2>
                <p style={{ fontSize: '0.85rem' }}>Your account ({currentUser.email_id}) has been locked</p>
                <button onClick={onBackToHome} style={{ backgroundColor: '#ef4444', color: '#fff', border: 'none', borderRadius: '6px', padding: '0.5rem 1rem', cursor: 'pointer', fontWeight: 'bold' }}>
                    Back
                </button>
            </div>
        );
    }

    return (
        <div>
            {/* Top Header */}
            <div style={{ backgroundColor: '#0f172a', border: '1px solid #334155', padding: '0.65rem 0.85rem', borderRadius: '8px', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <button onClick={onBackToHome} style={{ backgroundColor: '#1e293b', color: '#38bdf8', border: '1px solid #475569', borderRadius: '6px', padding: '0.3rem 0.6rem', cursor: 'pointer', fontWeight: '600', fontSize: '0.8rem' }}>
                        ←
                    </button>
                    <div>
                        <span style={{ fontWeight: '600', color: '#f8fafc', fontSize: '0.88rem' }}>👤 {currentUser.username}</span>
                    </div>
                </div>

                {/* 1-Minute Mute Timer Banner */}
                {muteSeconds > 0 && (
                    <div style={{ backgroundColor: '#78350f', border: '1px solid #f59e0b', color: '#fef3c7', padding: '0.2rem 0.6rem', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 'bold' }}>
                        🔇 CHAT FREEZED: {muteSeconds}s remaining
                    </div>
                )}

                <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '0.7rem', color: '#94a3b8', display: 'block' }}>Connected Friend:</span>
                    <span style={{ fontSize: '0.82rem', color: '#38bdf8', fontWeight: '600' }}>{peerUser ? peerUser.username : currentUser.friend_email}</span>
                </div>
            </div>

            {notification && (
                <div style={{ backgroundColor: '#450a0a', border: '1px solid #ef4444', color: '#fca5a5', padding: '0.55rem 0.85rem', borderRadius: '6px', fontSize: '0.8rem', marginBottom: '0.65rem', display: 'flex', justifyContent: 'space-between' }}>
                    <span>{notification.msg}</span>
                    <button onClick={() => setNotification(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}>✕</button>
                </div>
            )}

            {connectionStatus === 'FAILED' ? (
                <div style={{ height: '340px', backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '1.5rem', textAlign: 'center' }}>
                    <h3 style={{ color: '#f59e0b', margin: '0 0 0.4rem 0', fontSize: '1rem' }}>Connection Pending</h3>
                    <p style={{ color: '#94a3b8', fontSize: '0.82rem' }}>Waiting for friend ({currentUser.friend_email}) to enter your registered email address.</p>
                </div>
            ) : (
                <div>
                    <div ref={chatScrollRef} style={{ height: '340px', backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '0.85rem', overflowY: 'auto', marginBottom: '0.75rem' }}>
                        {messages.map((msg, idx) => {
                            const isMe = msg.sender_email === currentUser.email_id;
                            return (
                                <div key={msg.id || idx} style={{ textAlign: isMe ? 'right' : 'left', margin: '0.6rem 0' }}>
                                    <div style={{
                                        display: 'inline-block',
                                        padding: '0.55rem 0.85rem',
                                        borderRadius: isMe ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
                                        backgroundColor: isMe ? '#0284c7' : '#334155',
                                        color: '#f8fafc',
                                        maxWidth: '78%',
                                        textAlign: 'left'
                                    }}>
                                        <span style={{ display: 'block', fontSize: '0.68rem', color: isMe ? '#bae6fd' : '#94a3b8', fontWeight: '600', marginBottom: '0.15rem' }}>{msg.sender_username}</span>
                                        <span style={{ fontSize: '0.85rem', lineHeight: '1.3' }}>{msg.text}</span>
                                        <span style={{ display: 'block', fontSize: '0.62rem', color: isMe ? '#e0f2fe' : '#cbd5e1', marginTop: '0.25rem', textAlign: 'right' }}>{msg.timestamp}</span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <input
                            type="text"
                            placeholder={muteSeconds > 0 ? `Chat frozen due to high toxicity...` : "Type a message..."}
                            disabled={muteSeconds > 0}
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                            style={{ flex: 1, padding: '0.65rem 0.8rem', backgroundColor: '#0f172a', border: '1px solid #334155', color: '#f8fafc', borderRadius: '6px', fontSize: '0.85rem', outline: 'none' }}
                        />
                        <button onClick={sendMessage} disabled={muteSeconds > 0} style={{ padding: '0.65rem 1.2rem', backgroundColor: muteSeconds > 0 ? '#475569' : '#10b981', color: muteSeconds > 0 ? '#94a3b8' : '#022c22', border: 'none', borderRadius: '6px', fontWeight: '700', cursor: muteSeconds > 0 ? 'not-allowed' : 'pointer' }}>Send</button>
                    </div>
                </div>
            )}
        </div>
    );
}