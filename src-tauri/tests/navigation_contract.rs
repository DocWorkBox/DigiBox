use avtr1_desktop::navigation::{classify_navigation, NavigationDecision};

const SPLASH: &str = "tauri://localhost/splash.html";

#[test]
fn bundled_splash_must_match_the_exact_configured_url() {
    assert_eq!(
        classify_navigation(SPLASH, SPLASH),
        NavigationDecision::Splash
    );
    assert_eq!(
        classify_navigation("tauri://localhost/other.html", SPLASH),
        NavigationDecision::Deny
    );
    assert_eq!(
        classify_navigation("tauri://localhost/splash.html?remote=1", SPLASH),
        NavigationDecision::Deny
    );
}

#[test]
fn application_navigation_is_limited_to_the_exact_loopback_origin() {
    for allowed in [
        "http://127.0.0.1:7860/",
        "http://127.0.0.1:7860/settings?tab=audio#voice",
    ] {
        assert_eq!(
            classify_navigation(allowed, SPLASH),
            NavigationDecision::Application,
            "expected application URL: {allowed}"
        );
    }

    for denied in [
        "http://localhost:7860/",
        "http://127.0.0.1:8000/",
        "https://127.0.0.1:7860/",
        "http://[::1]:7860/",
        "http://127.0.0.1/",
    ] {
        assert_eq!(
            classify_navigation(denied, SPLASH),
            NavigationDecision::Deny,
            "expected denied URL: {denied}"
        );
    }
}

#[test]
fn only_allowlisted_https_hosts_can_open_externally() {
    for allowed in [
        "https://github.com/avaturn-live/avtr-1/",
        "https://huggingface.co/avaturn-live/avtr-1",
        "https://platform.qianwenai.com/docs/",
        "https://platform.minimaxi.com/docs/",
    ] {
        assert_eq!(
            classify_navigation(allowed, SPLASH),
            NavigationDecision::External,
            "expected external URL: {allowed}"
        );
    }

    for denied in [
        "http://github.com/avaturn-live/avtr-1/",
        "https://evil.github.com/",
        "https://github.com.evil.invalid/",
        "javascript:alert(1)",
        "not a URL",
    ] {
        assert_eq!(
            classify_navigation(denied, SPLASH),
            NavigationDecision::Deny,
            "expected denied URL: {denied}"
        );
    }
}
