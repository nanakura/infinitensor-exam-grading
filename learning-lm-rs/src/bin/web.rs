use learning_lm_rust::{chat, model};
use ntex::web::{self, HttpResponse};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::Tokenizer;

struct ChatManager {
    model: model::Llama<f32>,
    tokenizer: Tokenizer,
    sessions: HashMap<String, (chat::ChatSession, model::KVCache<f32>)>,
}

struct AppState {
    chat_manager: Mutex<ChatManager>,
}

#[derive(Deserialize)]
struct ChatRequest {
    session_id: String,
    message: String,
}

#[derive(Serialize)]
struct ChatResponse {
    message: String,
}

async fn init_chat(data: web::Data<AppState>, req: web::Json<ChatRequest>) -> HttpResponse {
    let mut manager = data.chat_manager.lock().unwrap();
    
    if !manager.sessions.contains_key(&req.session_id) {
        let mut session = chat::ChatSession::new();
        session.add_message("system", "You are a helpful assistant");
        let cache = manager.model.new_cache();
        manager.sessions.insert(req.session_id.clone(), (session, cache));
    }
    
    HttpResponse::Ok().json(ChatResponse {
        message: "Chat session initialized".to_string(),
    })
}

async fn send_message(data: web::Data<AppState>, req: web::Json<ChatRequest>) -> HttpResponse {
    let mut manager = data.chat_manager.lock().unwrap();
    
    if let Some((session, cache)) = manager.sessions.get_mut(&req.session_id) {
        session.add_message("user", &req.message);
        
        let prompt = session.format_prompt();
        let response = manager.model.chat(&prompt, cache, &manager.tokenizer, 500, 0.8, 30, 1.);
        
        session.add_message("assistant", &response);
        
        HttpResponse::Ok().json(ChatResponse { message: response })
    } else {
        HttpResponse::BadRequest().json(ChatResponse {
            message: "Session not found".to_string(),
        })
    }
}

async fn index() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html")
        .body(include_str!("../../static/dist/index.html"))
}

#[ntex::main]
async fn main() -> std::io::Result<()> {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let model = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let chat_manager = ChatManager {
        model,
        tokenizer,
        sessions: HashMap::new(),
    };
    
    let state = web::Data::new(AppState {
        chat_manager: Mutex::new(chat_manager),
    });

    println!("Starting server at http://127.0.0.1:8080");

    web::HttpServer::new(move || {
        web::App::new()
            .app_data(state.clone())
            .service(web::resource("/").to(index))
            .service(web::resource("/api/init").to(init_chat))
            .service(web::resource("/api/chat").to(send_message))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}