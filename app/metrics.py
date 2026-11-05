"""Prometheus metrics and request-timing middleware. Implemented in step 4.

Planned metrics:
  http_request_duration_seconds   histogram  {method, route, status}
  model_inference_seconds         histogram  {model}  (sklearn | llm)
  http_requests_in_flight         gauge
  llm_tokens_per_second           histogram
  inference_errors_total          counter    {model, reason}
"""
