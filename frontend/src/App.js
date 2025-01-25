import React, {useEffect, useRef, useState} from 'react';
import './App.css';

const App = () => {
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);
    const [chatHistory, setChatHistory] = useState([]);
    const [activeChatId, setActiveChatId] = useState(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
    };

    useEffect(() => {
        if (activeChatId !== null) {
            const activeChat = chatHistory.find(chat => chat.id === activeChatId);
            if (activeChat) {
                setMessages([
                    {id: Date.now(), type: 'user', content: activeChat.userQuery},
                    {id: Date.now() + 1, type: 'ai', content: activeChat.aiResponse, sources: activeChat.sources},
                ]);
            }
        }
    }, [activeChatId, chatHistory]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const query = inputValue.trim();

        if (!query) return;

        const newMessage = {id: Date.now(), type: 'user', content: query};
        setMessages((prev) => [...prev, newMessage]);

        setInputValue('');
        setIsLoading(true);

        try {
            const response = await fetch('http://0.0.0.0:8000/api/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: query}),
            });

            const data = await response.json();

            const aiMessage = {
                id: Date.now() + 1,
                type: 'ai',
                content: data.result,
                sources: data.sources || [],
            };
            setMessages((prev) => [...prev, aiMessage]);

            const newChat = {
                id: Date.now(),
                userQuery: query,
                aiResponse: data.result,
                sources: data.sources || [],
            };
            setChatHistory((prev) => [...prev, newChat]);
            setActiveChatId(newChat.id);
        } catch (error) {
            console.error('Query error:', error);
            setMessages((prev) => [
                ...prev,
                {
                    id: Date.now(),
                    type: 'error',
                    content: 'Something went wrong. Please try again.',
                },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleTabClick = (chatId) => {
        setActiveChatId(chatId);
    };

    const handleNewChat = () => {
        setActiveChatId(null);
        setMessages([]);
    };

    return (
        <div className="appContainer">
            <div className="sidebar">
                <h2>Chat History</h2>
                <div className="chatTabs">
                    {chatHistory.map((chat) => (
                        <div
                            key={chat.id}
                            className={`chatTab ${chat.id === activeChatId ? 'active' : ''}`}
                            onClick={() => handleTabClick(chat.id)}
                        >
                            <strong>{chat.userQuery ? chat.userQuery.substring(0, 20) : 'Untitled'}...</strong>
                        </div>
                    ))}
                </div>
                <button className="newChatButton" onClick={handleNewChat}>
                    Create New Chat
                </button>
            </div>

            <div className="chatContainer">
                <div className="chatHeader">
                    <h1>PubMed RAG</h1>
                </div>

                <div className="messagesContainer">
                    {messages.map((message) => (
                        <div
                            key={message.id}
                            className={`messageWrapper ${
                                message.type === 'user'
                                    ? 'userMessageAlign'
                                    : 'aiMessageAlign'
                            }`}
                        >
                            <div
                                className={`messageBox ${
                                    message.type === 'user'
                                        ? 'userMessage'
                                        : 'aiMessage'
                                }`}
                            >
                                {message.content}

                                {message.sources && message.sources.length > 0 && (
                                    <div className="sourceDetailsSidebar">
                                        <h3>Sources ({message.sources.length})</h3>
                                        {message.sources.map((source, index) => (
                                            <div key={index} className="sourceDetailsItem">
                                                {JSON.stringify(source)}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}

                    {isLoading && (
                        <div className="loadingIndicator">
                            <div className="loadingSpinner"></div>
                            <span>Searching research papers...</span>
                        </div>
                    )}

                    <div ref={messagesEndRef}/>
                </div>

                <form onSubmit={handleSubmit} className="inputContainer">
                    <input
                        ref={inputRef}
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        placeholder="Ask a research question..."
                        className="inputField"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        className="submitButton"
                        disabled={isLoading}
                    >
                        {isLoading ? '...' : 'Send'}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default App;
