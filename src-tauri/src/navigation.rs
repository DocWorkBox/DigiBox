use url::Url;

const EXTERNAL_HOSTS: &[&str] = &[
    "github.com",
    "huggingface.co",
    "platform.qianwenai.com",
    "platform.minimaxi.com",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NavigationDecision {
    Splash,
    Application,
    External,
    Deny,
}

pub fn classify_navigation(value: &str, bundled_splash_url: &str) -> NavigationDecision {
    if value == bundled_splash_url {
        return NavigationDecision::Splash;
    }

    let Ok(url) = Url::parse(value) else {
        return NavigationDecision::Deny;
    };
    if !url.username().is_empty() || url.password().is_some() {
        return NavigationDecision::Deny;
    }

    if url.scheme() == "http" && url.host_str() == Some("127.0.0.1") && url.port() == Some(7860) {
        return NavigationDecision::Application;
    }

    if url.scheme() == "https"
        && url
            .host_str()
            .is_some_and(|host| EXTERNAL_HOSTS.contains(&host))
    {
        return NavigationDecision::External;
    }

    NavigationDecision::Deny
}
