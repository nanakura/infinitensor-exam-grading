use futures::{pin_mut, StreamExt};
use learning_lm_rust::chat;
use learning_lm_rust::chat::ChatSession;
use learning_lm_rust::kvcache::KVCache;
#[cfg(not(feature = "cuda"))]
use learning_lm_rust::model::{self as model};
#[cfg(feature = "cuda")]
use learning_lm_rust::model_cuda::{self as model};
use learning_lm_rust::sse::Broadcaster;
use model::Llama;
use ntex::web::{self, HttpResponse};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Clone)]
struct ChatManager {
    model: Arc<Llama<f32>>,
    tokenizer: Tokenizer,
    sessions: HashMap<String, (ChatSession, KVCache<f32>)>,
}

struct AppState {
    chat_manager: Arc<Mutex<ChatManager>>,
    broadcaster: Arc<Mutex<Broadcaster>>,
}

#[derive(Serialize, Deserialize)]
pub struct ChatRequest {
    pub session_id: String,
    pub message: String,
}

#[derive(Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: String,
}

async fn init_chat(
    session_id: web::types::Path<String>,
    data: web::types::State<AppState>,
) -> HttpResponse {
    let mut manager = data.chat_manager.lock();

    let session_id_inner = session_id.into_inner();
    println!("init: session_id: {}", &session_id_inner);
    if !manager.sessions.contains_key(&session_id_inner) {
        let mut session = chat::ChatSession::new();
        session.add_message("system", "You are a helpful assistant");
        let cache = manager.model.new_cache();
        manager
            .sessions
            .insert(session_id_inner, (session, cache));
    }

    let rx = data.broadcaster.lock().new_client();

    HttpResponse::Ok()
        .header("content-type", "text/event-stream")
        .no_chunking()
        .streaming(rx)
}

async fn send_message(
    data: web::types::State<AppState>,
    req: web::types::Json<ChatRequest>,
) -> HttpResponse {
    let mut manager = data.chat_manager.lock();
    let session_id = &req.session_id;
    let message = &req.message;
    let payload = manager.sessions.get_mut(session_id);
    if let Some((session, cache)) = payload {
        session.add_message("user", message);

        let prompt = session.format_prompt();
        let manager = data.chat_manager.lock();
        let token_stream =
            manager
                .model
                .chat_stream(&prompt, cache, &manager.tokenizer, 500, 0.8, 30, 1.);
        pin_mut!(token_stream);
        while let Some(token) = token_stream.next().await {
            let response = token.to_string();
            session.add_message("assistant", &response);
            data.broadcaster.lock().send(&response);

            session.add_message("assistant", &response);
        }

        HttpResponse::Ok().json(&ChatResponse {
            message: "msg sent".to_string(),
        })
    } else {
        HttpResponse::BadRequest().json(&ChatResponse {
            message: "Session not found".to_string(),
        })
    }
}

// async fn index() -> HttpResponse {
//     HttpResponse::Ok()
//         .content_type("text/html")
//         .body(include_str!("../static/index.html"))
// }

#[ntex::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let model = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let chat_manager = ChatManager {
        model: Arc::new(model),
        tokenizer,
        sessions: HashMap::new(),
    };

    let broadcaster = Broadcaster::create();

    println!("Starting server at http://127.0.0.1:8080");

    web::HttpServer::new(move || {
        web::App::new()
            .wrap(web::middleware::Logger::default())
            .wrap(ntex_cors::Cors::default())
            .state(AppState {
                chat_manager: Arc::new(Mutex::new(chat_manager.clone())),
                broadcaster: broadcaster.clone(),
            })
            //.service(web::resource("/").to(index))
            .route("/api/init/{session_id}", web::get().to(init_chat))
            .route("/api/chat", web::post().to(send_message))
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}
