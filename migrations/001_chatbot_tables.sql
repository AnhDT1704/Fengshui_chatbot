-- 001_chatbot_tables.sql
-- Two operational tables for the LangGraph chatbot.
--   conversation_log : per-turn log used for memory + analytics
--   escalation_queue : items written by escalate_to_human_tool

CREATE TABLE IF NOT EXISTS conversation_log (
    id            BIGSERIAL    PRIMARY KEY,
    session_id    VARCHAR(100) NOT NULL,
    role          VARCHAR(10)  NOT NULL,
    content       TEXT         NOT NULL,
    agent_used    VARCHAR(50),
    intent        VARCHAR(100),
    tools_called  VARCHAR[],
    created_at    TIMESTAMP    DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conv_session_created
    ON conversation_log (session_id, created_at);

CREATE TABLE IF NOT EXISTS escalation_queue (
    id            BIGSERIAL    PRIMARY KEY,
    session_id    VARCHAR(100),
    reason        VARCHAR(50)  NOT NULL,
    user_summary  TEXT,
    full_context  TEXT,
    status        VARCHAR(20)  DEFAULT 'pending',
    created_at    TIMESTAMP    DEFAULT now(),
    resolved_at   TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_escalation_status_created
    ON escalation_queue (status, created_at);
