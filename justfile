set dotenv-load

default:
  @just --list --justfile {{justfile()}}

run:
  cargo r

build-release:
  cargo build --release --no-default-features

metal-hud:
  MTL_HUD_ENABLED=1 just run

build-wasm:
  cargo build --target wasm32-unknown-unknown --no-default-features
  wasm-bindgen --out-name bevy_simple_terrain --out-dir wasm --target web target/wasm32-unknown-unknown/debug/bevy_simple_terrain.wasm

build-wasm-release:
  cargo build --profile wasm-release --target wasm32-unknown-unknown --no-default-features
  wasm-bindgen --out-name bevy_simple_terrain --out-dir wasm --target web target/wasm32-unknown-unknown/wasm-release/bevy_simple_terrain.wasm

wasm-host:
  python3 -m http.server
