from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine


# =========================
# 1. Configuration
# =========================

SERVICE_ACCOUNT_JSON = "/home/hungphd/hung-gemini-project-0c4f3f87a126.json"

PROJECT_ID = "hung-gemini-project"
LOCATION = "global"   # examples: "global", "us", "eu"
ENGINE_ID = "generativeai_1777581059948"  # also called App ID in Agent Builder

QUERY = "What are the top-rated movies released in 2026?"


# =========================
# 2. Query-answer function
# =========================

def answer_query(
    service_account_json: str,
    project_id: str,
    location: str,
    engine_id: str,
    query: str,
):
    # Regional endpoint
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create client using service account JSON
    client = discoveryengine.ConversationalSearchServiceClient.from_service_account_file(
        service_account_json,
        client_options=client_options,
    )

    # Serving config path
    serving_config = (
        f"projects/{project_id}/locations/{location}/collections/default_collection/"
        f"engines/{engine_id}/servingConfigs/default_serving_config"
    )

    # Build request
    request = discoveryengine.AnswerQueryRequest(
        serving_config=serving_config,
        query=discoveryengine.Query(text=query),
        answer_generation_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
            include_citations=True,
            answer_language_code="en",
        ),
    )

    # Send request
    response = client.answer_query(request=request)

    return response


# =========================
# 3. Run
# =========================

if __name__ == "__main__":
    response = answer_query(
        service_account_json=SERVICE_ACCOUNT_JSON,
        project_id=PROJECT_ID,
        location=LOCATION,
        engine_id=ENGINE_ID,
        query=QUERY,
    )

    print("Question:")
    print(QUERY)

    print("\nAnswer:")
    print(response.answer.answer_text)

    print("\nFull response:")
    print(response)