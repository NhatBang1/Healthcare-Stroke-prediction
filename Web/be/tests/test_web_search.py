from app.chat.web_search import (
    search_stroke_education_images,
    search_stroke_web_sources,
    search_risk_factor_sources,
    should_search_stroke_images,
    should_search_stroke_web,
)


def test_stroke_image_search_is_scoped_to_symptom_education_terms():
    assert should_search_stroke_images("Mình hay đau đầu") is True
    assert should_search_stroke_images("Tôi muốn cập nhật BMI") is False


def test_tavily_tool_falls_back_to_no_media_without_api_key(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", None)

    assert search_stroke_education_images.invoke({"query": "stroke FAST"}) == []


def test_article_questions_trigger_web_source_search():
    assert should_search_stroke_web("Có bài báo nào khẳng định dấu hiệu này không?") is True
    assert should_search_stroke_web("Tôi muốn cập nhật BMI") is False
    assert should_search_stroke_web("hút thuốc, uống rượu có ảnh hưởng đến nguy cơ đột quỵ không?") is True


def test_web_source_tool_falls_back_to_no_sources_without_api_key(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", None)

    assert search_stroke_web_sources.invoke({"query": "stroke headache evidence"}) == []


def test_risk_factor_sources_only_search_when_explicit_factors_exist(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", None)

    assert search_risk_factor_sources({"smoke_flag": True, "alco_flag": True}) == []
