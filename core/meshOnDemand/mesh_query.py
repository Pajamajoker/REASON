import requests
from pathlib import Path
import json
from core.meshOnDemand.mesh_utils.mesh_response_parser import extract_from_pre_block, build_mesh_query
from core.pubmedSearch.pubmedApiSearch import query_pubmed_api
json_query = {
  "qid": "6d5487fd37ad",
  "created_at": "2025-10-07T00:32:09Z",
  "pico_valid": {
    "Population": "adults with septic shock",
    "Intervention": "early norepinephrine",
    "Comparator": "dopamine",
    "Outcomes": [
      "28-day mortality"
    ]
  },
  "augmented": {
    "Population": {
      "value": "adults with septic shock",
      "synonyms": [
        "patients with septic shock",
        "individuals with septic shock",
        "patients-with-septic-shock",
        "individuals-with-septic-shock"
      ]
    },
    "Intervention": {
      "value": "early norepinephrine",
      "synonyms": [
        "norepinephrine",
        "norepinephrines",
        "early-norepinephrine"
      ]
    },
    "Comparator": {
      "value": "dopamine",
      "synonyms": [
        "dopamine treatment",
        "dopamines",
        "dopamine-treatment"
      ]
    },
    "Outcomes": [
      {
        "value": "28-day mortality",
        "synonyms": [
          "28-day mortality rate",
          "28 day mortality rate",
          "28-day-mortality-rate"
        ]
      }
    ]
  },
  "model": "gemma-3 via Vertex",
  "temperature": 0.2
}

def query_mesh_api(pico_json=json_query):
    """
    Queries the Mesh API with the given Pico query and returns the response.

    Args:
        pico_json (dict): The Pico query JSON.

    Returns:
        dict: The txt response from the Mesh API.
    """
    headers = {"Content-Type": "application/json"}

    response = requests.post(
        "https://meshb.nlm.nih.gov/api/MOD",
        headers=headers,
        json={"input": json.dumps(pico_json['pico_valid'])},
    )


    if response.status_code == 200:
        base = Path(__file__).resolve().parent
        (base.parent / "artifacts_day3" / f"mesh_{pico_json['qid']}.txt").write_text(response.text, encoding="utf-8")
        mesh_response = extract_from_pre_block(response.text)
        mesh_response['qid'] = pico_json['qid']
        (base.parent / "artifacts_day3" / f"mesh_parsed{pico_json['qid']}.json").write_text(json.dumps(mesh_response, ensure_ascii=False, indent=2), encoding="utf-8")

        mesh_query = build_mesh_query(mesh_response['mesh_terms']+mesh_response['relevant_mesh_terms'])
        print(mesh_query)
        temp_dict = {"qid": pico_json['qid'], "mesh_query": mesh_query}
        with open(base.parent / "artifacts_day3" / "mesh_query.jsonl", "a", encoding="utf-8") as f:
          f.write(json.dumps(temp_dict, ensure_ascii=False) + "\n")  

        pubmed_response = query_pubmed_api(mesh_query)
        pubmed_response['qid'] = pico_json['qid']
        (base.parent / "artifacts_day3" / f"pubmed_parsed{pico_json['qid']}.json").write_text(
            json.dumps(pubmed_response, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        return response.text
    else:
        response.raise_for_status()

if __name__ == "__main__":
    result = query_mesh_api()
    # print(result)