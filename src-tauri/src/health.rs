use std::time::Duration;

use serde::Deserialize;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HealthFailure {
    RequestFailed(String),
    HttpStatus(u16),
    InvalidJson(String),
    IdentityMismatch,
    ServiceStatus(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HealthProbe {
    pub healthy: bool,
    pub failure: Option<HealthFailure>,
    pub http_status: Option<u16>,
    pub service: Option<String>,
    pub service_status: Option<String>,
    pub instance_id: Option<String>,
}

impl HealthProbe {
    fn failed(failure: HealthFailure, http_status: Option<u16>) -> Self {
        Self {
            healthy: false,
            failure: Some(failure),
            http_status,
            service: None,
            service_status: None,
            instance_id: None,
        }
    }

    pub fn matches_owned_instance(&self, expected_instance_id: &str) -> bool {
        self.healthy && self.instance_id.as_deref() == Some(expected_instance_id)
    }
}

#[derive(Deserialize)]
struct HealthBody {
    service: Option<String>,
    status: Option<String>,
    instance_id: Option<String>,
}

pub async fn probe_avtr_service(
    client: &reqwest::Client,
    url: &str,
    request_timeout: Duration,
) -> HealthProbe {
    let response = match client.get(url).timeout(request_timeout).send().await {
        Ok(response) => response,
        Err(error) => {
            return HealthProbe::failed(HealthFailure::RequestFailed(error.to_string()), None);
        }
    };
    let status_code = response.status();
    if !status_code.is_success() {
        return HealthProbe::failed(
            HealthFailure::HttpStatus(status_code.as_u16()),
            Some(status_code.as_u16()),
        );
    }

    let body = match response.json::<HealthBody>().await {
        Ok(body) => body,
        Err(error) => {
            return HealthProbe::failed(
                HealthFailure::InvalidJson(error.to_string()),
                Some(status_code.as_u16()),
            );
        }
    };
    if body.service.as_deref() != Some("avtr1-streamer") {
        return HealthProbe {
            healthy: false,
            failure: Some(HealthFailure::IdentityMismatch),
            http_status: Some(status_code.as_u16()),
            service: body.service,
            service_status: body.status,
            instance_id: body.instance_id,
        };
    }

    let healthy = matches!(body.status.as_deref(), Some("ready" | "degraded"));
    let failure = if healthy {
        None
    } else {
        Some(HealthFailure::ServiceStatus(
            body.status.clone().unwrap_or_else(|| "unknown".to_owned()),
        ))
    };
    HealthProbe {
        healthy,
        failure,
        http_status: Some(status_code.as_u16()),
        service: body.service,
        service_status: body.status,
        instance_id: body.instance_id,
    }
}
