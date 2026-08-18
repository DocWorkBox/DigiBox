use std::time::Duration;

use avtr1_desktop::app::{probe_app_root, AppRootFailure};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

async fn serve_once(status: u16, reason: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 1024];
        let size = stream.read(&mut request).await.unwrap();
        let request = String::from_utf8_lossy(&request[..size]);
        assert!(request.starts_with("GET / HTTP/1.1"));
        let response =
            format!("HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    format!("http://{address}/")
}

#[tokio::test]
async fn app_root_probe_requires_a_successful_root_document() {
    let client = reqwest::Client::new();
    let ready_url = serve_once(200, "OK").await;
    let unavailable_url = serve_once(503, "Service Unavailable").await;

    assert_eq!(
        probe_app_root(&client, &ready_url, Duration::from_secs(1)).await,
        Ok(())
    );
    assert_eq!(
        probe_app_root(&client, &unavailable_url, Duration::from_secs(1)).await,
        Err(AppRootFailure::HttpStatus(503))
    );
}

#[tokio::test]
async fn app_root_probe_reports_transport_failure() {
    let client = reqwest::Client::new();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);

    assert!(matches!(
        probe_app_root(
            &client,
            &format!("http://{address}/"),
            Duration::from_millis(250)
        )
        .await,
        Err(AppRootFailure::RequestFailed(_))
    ));
}
