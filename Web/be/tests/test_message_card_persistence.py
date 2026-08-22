import json

from app.memory import MemorySessionStore, RedisSessionStore
from app.persistence.conversation_store import PostgresConversationStore


def test_memory_store_round_trips_assistant_card():
    store = MemorySessionStore()
    card = {"model_used": "stroke_model_clinical.joblib", "risk_label": "medium_risk"}

    store.add_turn("card-session", "Thông tin của tôi", "Kết quả", assistant_card=card)

    messages = store.messages("card-session")
    assert messages[0]["id"]
    assert messages[1]["id"]
    assert messages[0]["id"] != messages[1]["id"]
    assert messages[0]["role"] == "user"
    assert messages[1]["assistant_card"] == card


def test_memory_store_round_trips_reply_context_on_user_message():
    store = MemorySessionStore()
    reply_context = {
        "message_id": "assistant-message-1",
        "quote": "Đây là đoạn trả lời được chọn.",
    }

    store.add_turn(
        "reply-session",
        "Bạn giải thích kỹ hơn được không?",
        "Được.",
        reply_context=reply_context,
    )

    assert store.messages("reply-session")[0]["reply_context"] == reply_context


def test_memory_store_round_trips_assistant_media():
    store = MemorySessionStore()
    media = [{"type": "image", "url": "https://example.org/fast.png", "title": "FAST"}]

    store.add_turn(
        "media-session",
        "Hay đau đầu",
        "Tư vấn triệu chứng",
        media=media,
    )

    assert store.messages("media-session")[1]["media"] == media


def test_postgres_message_serializer_exposes_reply_context_metadata():
    message = PostgresConversationStore._serialize_message(
        {
            "id": "user-message-1",
            "role": "user",
            "content": "Giải thích thêm",
            "metadata": {
                "reply_context": {
                    "message_id": "assistant-message-1",
                    "quote": "Đoạn được chọn",
                }
            },
            "created_at": None,
        }
    )

    assert message["reply_context"] == {
        "message_id": "assistant-message-1",
        "quote": "Đoạn được chọn",
    }


def test_redis_store_normalizes_legacy_ids_and_reply_context():
    class FakeRedis:
        def execute(self, command, *args):
            assert command == "LRANGE"
            return [
                json.dumps(
                    {
                        "role": "user",
                        "content": "Giải thích thêm",
                        "reply_context": {
                            "message_id": "assistant-legacy",
                            "quote": "Đoạn được chọn",
                        },
                    }
                )
            ]

    store = RedisSessionStore.__new__(RedisSessionStore)
    store._redis = FakeRedis()

    first = store.messages("redis-reply-session")[0]
    second = store.messages("redis-reply-session")[0]

    assert first["id"].startswith("legacy-")
    assert first["id"] == second["id"]
    assert first["reply_context"]["quote"] == "Đoạn được chọn"
