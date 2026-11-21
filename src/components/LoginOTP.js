import React, { useState } from 'react';
import axios from 'axios';

const API_BASE = "http://localhost:8000/api";

export default function LoginOTP({ onLoginSuccess, windowName }) {
    const [username, setUsername] = useState('');
    const [mobile, setMobile] = useState('');
    const [email, setEmail] = useState('');
    const [friendEmail, setFriendEmail] = useState('');
    const [otp, setOtp] = useState('');
    const [step, setStep] = useState(1);
    const [loading, setLoading] = useState(false);
    const [notification, setNotification] = useState(null);

    const showNotice = (msg, type = 'error') => {
        setNotification({ msg, type });
    };

    // 1. Sanitized Input Handlers
    const handleUsernameChange = (e) => {
        let val = e.target.value;
        // Strip leading spaces
        val = val.replace(/^\s+/, '');
        // Allow ONLY alphabets and internal spaces
        val = val.replace(/[^a-zA-Z\s]/g, '');
        setUsername(val);
    };

    const handleMobileChange = (e) => {
        let val = e.target.value;
        // Allow ONLY digits
        val = val.replace(/\D/g, '');
        // Enforce maximum 10 digits
        if (val.length > 10) {
            val = val.slice(0, 10);
        }
        setMobile(val);
    };

    const handleEmailChange = (val, setter) => {
        // Strip all whitespace
        const cleanVal = val.replace(/\s/g, '');
        setter(cleanVal);
    };

    // 2. Comprehensive Pre-Submission Validation
    const validateInputs = () => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

        if (!username.trim()) {
            showNotice("Please enter your username.");
            return false;
        }
        if (username.trim().length < 2) {
            showNotice("Username must be at least 2 characters.");
            return false;
        }

        if (!mobile.trim() || mobile.length !== 10) {
            showNotice("Please enter a valid 10-digit mobile number.");
            return false;
        }

        if (!email.trim() || !email.includes('@') || !emailRegex.test(email)) {
            showNotice("Please enter a valid email address containing @.");
            return false;
        }

        if (!friendEmail.trim() || !friendEmail.includes('@') || !emailRegex.test(friendEmail)) {
            showNotice("Please enter a valid friend email address containing @.");
            return false;
        }

        if (email.trim().toLowerCase() === friendEmail.trim().toLowerCase()) {
            showNotice("Your Email ID and Friend's Email ID cannot be identical.");
            return false;
        }

        return true;
    };

    const requestOTP = async () => {
        setNotification(null);
        if (!validateInputs()) return;

        setLoading(true);
        try {
            await axios.post(`${API_BASE}/auth/send_otp`, {
                username: username.trim(),
                mobile_number: mobile,
                email_id: email.trim(),
                friend_email: friendEmail.trim()
            });
            setStep(2);
            showNotice(`OTP dispatched to ${email}. Please check your inbox.`, 'success');
        } catch (err) {
            showNotice("Access Denied: " + (err.response?.data?.detail || err.message));
        }
        setLoading(false);
    };

    const verifyOTP = async () => {
        setNotification(null);
        if (!otp.trim()) {
            showNotice("Please enter the OTP sent to your email.");
            return;
        }

        setLoading(true);
        try {
            const res = await axios.post(`${API_BASE}/auth/verify_otp`, { mobile_number: mobile, otp: otp.trim() });
            if (res.data.status === "Authenticated") {
                onLoginSuccess({
                    username: res.data.username,
                    mobile_number: res.data.mobile_number,
                    email_id: res.data.email_id,
                    friend_email: res.data.friend_email
                });
            }
        } catch (err) {
            showNotice("Authentication Failed: " + (err.response?.data?.detail || "Invalid OTP entered."));
        }
        setLoading(false);
    };

    return (
        <div style={{ backgroundColor: '#0f172a', padding: '1.25rem', borderRadius: '8px', border: '1px solid #334155' }}>
            <h3 style={{ marginTop: 0, color: '#f8fafc', fontSize: '0.95rem', fontWeight: '600', marginBottom: '1rem' }}>
                🔐 Login Portal
            </h3>

            {/* Banner Notification */}
            {notification && (
                <div style={{
                    backgroundColor: notification.type === 'error' ? '#450a0a' : '#064e3b',
                    border: `1px solid ${notification.type === 'error' ? '#ef4444' : '#10b981'}`,
                    color: notification.type === 'error' ? '#fca5a5' : '#a7f3d0',
                    padding: '0.6rem 0.8rem',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    marginBottom: '1rem',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                }}>
                    <span>{notification.msg}</span>
                    <button onClick={() => setNotification(null)} style={{ background: 'none', border: 'none', color: 'inherit', fontWeight: 'bold', cursor: 'pointer' }}>✕</button>
                </div>
            )}

            {step === 1 ? (
                <div>
                    <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.8rem', fontWeight: '500', marginBottom: '0.25rem' }}>Username:</label>
                    <input
                        type="text"
                        placeholder="e.g. Sarayu"
                        value={username}
                        onChange={handleUsernameChange}
                        style={{ width: '92%', padding: '0.55rem 0.7rem', backgroundColor: '#1e293b', border: '1px solid #475569', color: '#fff', borderRadius: '6px', fontSize: '0.85rem', marginBottom: '0.75rem', outline: 'none' }}
                    />

                    <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.8rem', fontWeight: '500', marginBottom: '0.25rem' }}>Mobile Number:</label>
                    <input
                        type="text"
                        placeholder="e.g. 9876543210"
                        value={mobile}
                        onChange={handleMobileChange}
                        maxLength={10}
                        style={{ width: '92%', padding: '0.55rem 0.7rem', backgroundColor: '#1e293b', border: '1px solid #475569', color: '#fff', borderRadius: '6px', fontSize: '0.85rem', marginBottom: '0.75rem', outline: 'none' }}
                    />

                    <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.8rem', fontWeight: '500', marginBottom: '0.25rem' }}>Your Registered Email ID:</label>
                    <input
                        type="email"
                        placeholder="e.g. user1@gmail.com"
                        value={email}
                        onChange={(e) => handleEmailChange(e.target.value, setEmail)}
                        style={{ width: '92%', padding: '0.55rem 0.7rem', backgroundColor: '#1e293b', border: '1px solid #475569', color: '#fff', borderRadius: '6px', fontSize: '0.85rem', marginBottom: '0.75rem', outline: 'none' }}
                    />

                    <label style={{ display: 'block', color: '#38bdf8', fontSize: '0.8rem', fontWeight: '600', marginBottom: '0.25rem' }}>Friend's Email ID (To Connect With):</label>
                    <input
                        type="email"
                        placeholder="e.g. user2@gmail.com"
                        value={friendEmail}
                        onChange={(e) => handleEmailChange(e.target.value, setFriendEmail)}
                        style={{ width: '92%', padding: '0.55rem 0.7rem', backgroundColor: '#1e293b', border: '1px solid #0284c7', color: '#fff', borderRadius: '6px', fontSize: '0.85rem', marginBottom: '1.1rem', outline: 'none' }}
                    />

                    <button
                        onClick={requestOTP}
                        disabled={loading}
                        style={{ width: '100%', padding: '0.65rem', backgroundColor: '#10b981', color: '#022c22', border: 'none', borderRadius: '6px', fontWeight: '700', fontSize: '0.85rem', cursor: 'pointer' }}>
                        {loading ? "Sending..." : "Send Verification OTP via Email"}
                    </button>
                </div>
            ) : (
                <div>
                    <label style={{ display: 'block', color: '#94a3b8', marginBottom: '0.4rem', fontSize: '0.8rem' }}>Enter 6-Digit OTP:</label>
                    <input
                        type="text"
                        placeholder="Enter OTP"
                        value={otp}
                        onChange={(e) => setOtp(e.target.value)}
                        style={{ width: '92%', padding: '0.6rem 0.7rem', backgroundColor: '#1e293b', border: '1px solid #0284c7', color: '#38bdf8', borderRadius: '6px', fontSize: '0.95rem', letterSpacing: '2px', textAlign: 'center', marginBottom: '1rem', outline: 'none' }}
                    />
                    <button
                        onClick={verifyOTP}
                        disabled={loading}
                        style={{ width: '100%', padding: '0.65rem', backgroundColor: '#10b981', color: '#022c22', border: 'none', borderRadius: '6px', fontWeight: '700', fontSize: '0.85rem', cursor: 'pointer' }}>
                        {loading ? "Verifying..." : "Verify OTP & Login"}
                    </button>
                </div>
            )}
        </div>
    );
}
