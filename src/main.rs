#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use bevy::{
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        settings::{WgpuFeatures, WgpuSettings},
        RenderPlugin,
    },
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use noise::{Billow, Fbm, NoiseFn, Perlin, Seedable, Simplex, SuperSimplex, Worley};
use rand::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(RenderPlugin {
            wgpu_settings: WgpuSettings {
                features: WgpuFeatures::POLYGON_MODE_LINE,
                ..default()
            },
        }))
        // .add_plugin(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
        // .add_plugin(bevy::diagnostic::LogDiagnosticsPlugin::default())
        // .add_plugin(bevy::diagnostic::SystemInformationDiagnosticsPlugin::default())
        .add_plugin(WireframePlugin)
        .add_plugin(EguiPlugin)
        .insert_resource(NoiseConfig {
            scale: 0.8,
            magnitude: 0.3,
            kind: NoiseType::Perlin,
            x_offset: 0.,
            time_scale: 1.,
            plane_subdivisions: 100,
        })
        .add_startup_system(setup)
        .add_system(modify_terrain)
        .run();
}

#[derive(Component)]
struct Terrain;

#[derive(Resource)]
struct NoiseConfig {
    scale: f32,
    magnitude: f32,
    kind: NoiseType,
    x_offset: f32,
    time_scale: f32,
    plane_subdivisions: u32,
}

#[derive(PartialEq)]
enum NoiseType {
    Perlin,
    Simplex,
    SuperSimplex,
    Worley,
    Fbm,
}

#[derive(Resource)]
struct NoiseTexture(Handle<Image>);

fn modify_terrain(
    mut commands: Commands,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(&Handle<Mesh>, Entity, Option<&Wireframe>), With<Terrain>>,
    mut contexts: EguiContexts,
    mut first: Local<bool>,
    mut noise_config: ResMut<NoiseConfig>,
    nt: Res<NoiseTexture>,
    mut images: ResMut<Assets<Image>>,
) {
    // let mut rng = rand::thread_rng();

    let mut changed = false;

    egui::Window::new("Noise").show(contexts.ctx_mut(), |ui| {
        // egui::Ui::radio_value()

        ui.radio_value(&mut noise_config.kind, NoiseType::Perlin, "Perlin");
        ui.radio_value(&mut noise_config.kind, NoiseType::Simplex, "Simplex");
        ui.radio_value(
            &mut noise_config.kind,
            NoiseType::SuperSimplex,
            "SuperSimple",
        );
        ui.radio_value(&mut noise_config.kind, NoiseType::Worley, "Worley");
        ui.radio_value(&mut noise_config.kind, NoiseType::Fbm, "Fbm");

        ui.horizontal(|ui| {
            ui.label("Scale");
            if ui
                .add(egui::Slider::new(&mut noise_config.scale, 0.1f32..=10.0f32).logarithmic(true))
                .changed()
            {
                changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Magnitude");
            if ui
                .add(egui::Slider::new(
                    &mut noise_config.magnitude,
                    0.0f32..=1.0f32,
                ))
                .changed()
            {
                changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Time Scale");
            if ui
                .add(
                    egui::Slider::new(&mut noise_config.time_scale, 0.0f32..=19.0f32)
                        .logarithmic(true),
                )
                .changed()
            {
                changed = true;
            }
        });

        if ui.button("Toggle Wireframe").clicked() {
            query.iter().for_each(|(_, e, w)| {
                if w.is_some() {
                    commands.entity(e).remove::<Wireframe>();
                } else {
                    commands.entity(e).insert(Wireframe);
                }
            });
        }

        ui.horizontal(|ui| {
            ui.label("Plane Subdivisions");
            if ui
                .add(egui::Slider::new(
                    &mut noise_config.plane_subdivisions,
                    0..=1000,
                ))
                .changed()
            {
                query.iter().for_each(|(_q, e, _)| {
                    let meesh = meshes.add(Mesh::from(shape::Plane {
                        size: 10.,
                        subdivisions: noise_config.plane_subdivisions,
                    }));
                    commands.entity(e).remove::<Handle<Mesh>>().insert(meesh);
                });
            }
        });

        // let mut _scale = *scale;
        // ui.add(egui::Slider::new(&mut _scale, 0.1f32..=1.0f32));
        // *scale = _scale;
    });

    noise_config.x_offset += time.delta_seconds() * noise_config.time_scale;

    // if !changed && *first {
    //     return;
    // }

    // let p = Perlin::new(1);
    // let s = Simplex::new(1);
    let noise_fn: Box<dyn NoiseFn<f64, 2>> = match noise_config.kind {
        NoiseType::Perlin => Box::new(Perlin::new(1)),
        NoiseType::Simplex => Box::new(Simplex::new(1)),
        NoiseType::SuperSimplex => Box::new(SuperSimplex::new(1)),
        // NoiseType::Billow => Box::new(Billow::new(1)),
        NoiseType::Worley => Box::new(Worley::new(1)),
        NoiseType::Fbm => Box::new(Fbm::<Perlin>::new(1)),
    };

    let noise_texture = images.get_mut(&nt.0).unwrap();

    let dim = 256;

    noise_texture
        .data
        .chunks_exact_mut(4)
        .enumerate()
        .for_each(|(i, c)| {
            let x = (i % dim) as f32;
            let y = (i / dim) as f32;
            let v = noise_fn.get([
                ((noise_config.x_offset + x) * noise_config.scale) as f64,
                (y * noise_config.scale) as f64,
            ]) as f32
                * noise_config.magnitude;

            // shift 0.0..1.0 to 0..255
            let m = 255.;

            c[0] = (v * m) as u8;
            c[1] = (v * m) as u8;
            c[2] = (v * m) as u8;
            c[3] = 255;
        });

    query.iter().for_each(|(h, _, _)| {
        if let Some(m) = meshes.get_mut(h) {
            if !*first {
                *first = true;
            }

            if let Some(bevy::render::mesh::VertexAttributeValues::Float32x3(ref mut v)) =
                m.attribute_mut(Mesh::ATTRIBUTE_POSITION)
            {
                v.iter_mut().for_each(|v| {
                    v[1] = noise_fn.get([
                        ((noise_config.x_offset + v[0]) * noise_config.scale) as f64,
                        (v[2] * noise_config.scale) as f64,
                    ]) as f32
                        * noise_config.magnitude;
                });
            }
            compute_normals(m);
        }
    });
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    noise_config: Res<NoiseConfig>,
) {
    // commands.spawn(SpriteBundle {
    //     texture: asset_server.load("sandwich.png"),
    //     ..default()
    // });

    let width = 256;
    let height = 256;

    let img_data = vec![u8::MAX; width * height * 4];

    let gen_img = Image::new(
        Extent3d {
            width: width as u32,
            height: height as u32,
            ..Default::default()
        },
        TextureDimension::D2,
        img_data,
        TextureFormat::Rgba8Unorm,
    );

    let h = images.add(gen_img);

    commands.insert_resource(NoiseTexture(h.clone()));

    commands.spawn(ImageBundle {
        image: h.into(),
        style: Style {
            // align_content: AlignContent::Center,
            // align_items: AlignItems::Center,
            // display: Display::Flex,
            // flex_direction: FlexDirection::Row,
            // justify_content: JustifyContent::Center,
            size: Size::new(Val::Px(width as f32), Val::Px(height as f32)),
            ..default()
        },
        ..default()
    });

    // plane
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(
                shape::Plane {
                    size: 10.,
                    subdivisions: noise_config.plane_subdivisions,
                }
                .into(),
            ),
            material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
            // transform: Transform::from_xyz(0., 0., 0.),
            ..default()
        },
        Terrain,
        // Wireframe,
    ));

    // commands.spawn(PbrBundle {
    //     mesh: meshes.add(build_terrain()),
    //     material: materials.add(Color::rgb(0.9, 0.5, 0.3).into()),
    //     ..default()
    // });

    // cube
    // commands.spawn(PbrBundle {
    //     mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
    //     material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
    //     transform: Transform::from_xyz(0.0, 0.5, 0.0),
    //     ..default()
    // });

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 4.0, 0.0),
        // transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // commands.spawn(DirectionalLightBundle {
    //     transform: Transform::from_xyz(40.0, 8.0, 40.0).looking_at(Vec3::ZERO, Vec3::Y),
    //     directional_light: DirectionalLight { ..default() },
    //     ..default()
    // });

    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

// fn build_terrain() -> Mesh {
//     let mut positions: Vec<[f32; 3]> = Vec::new();
//     let mut normals: Vec<[f32; 3]> = Vec::new();
//     let mut uvs: Vec<[f32; 2]> = Vec::new();
//     let mut indices: Vec<u32> = Vec::new();
//     let up = Vec3::Y.to_array();

//     positions.push([0., 0., 0.]);
//     normals.push(up);
//     uvs.push([0., 0.]);

//     positions.push([0., 0., 1.]);
//     normals.push(up);
//     uvs.push([0., 1.]);

//     positions.push([1., 0., 0.]);
//     normals.push(up);
//     uvs.push([1., 0.]);

//     indices.push(0);
//     indices.push(1);
//     indices.push(2);

//     let mut mesh = Mesh::new(bevy::render::render_resource::PrimitiveTopology::TriangleList);
//     mesh.set_indices(Some(bevy::render::mesh::Indices::U32(indices)));
//     mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
//     mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
//     mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
//     mesh
// }

fn compute_normals(msh: &mut Mesh) {
    #[inline]
    fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
        let (a, b, c) = (Vec3::from(a), Vec3::from(b), Vec3::from(c));
        (b - a).cross(c - a).normalize().into()
    }

    assert!(
        matches!(
            msh.primitive_topology(),
            bevy::render::render_resource::PrimitiveTopology::TriangleList
        ),
        "`compute_normals` can only work on `TriangleList`s"
    );

    let positions = msh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .unwrap()
        .as_float3()
        .expect("`Mesh::ATTRIBUTE_POSITION` vertex attributes should be of type `float3`");

    match msh.indices() {
        Some(indices) => {
            let mut count: usize = 0;
            let mut corners = [0_usize; 3];
            let mut normals = vec![[0.0f32; 3]; positions.len()];
            let mut adjacency_counts = vec![0_usize; positions.len()];

            for i in indices.iter() {
                corners[count % 3] = i;
                count += 1;
                if count % 3 == 0 {
                    let normal = face_normal(
                        positions[corners[0]],
                        positions[corners[1]],
                        positions[corners[2]],
                    );
                    for corner in corners {
                        normals[corner] = (Vec3::from(normal) + Vec3::from(normals[corner])).into();
                        adjacency_counts[corner] += 1;
                    }
                }
            }

            // average (smooth) normals for shared vertices...
            // TODO: support different methods of weighting the average
            for i in 0..normals.len() {
                let count = adjacency_counts[i];
                if count > 0 {
                    normals[i] = (Vec3::from(normals[i]) / (count as f32)).normalize().into();
                }
            }

            msh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        }
        None => {
            let normals: Vec<_> = positions
                .chunks_exact(3)
                .map(|p| face_normal(p[0], p[1], p[2]))
                .flat_map(|normal| [normal; 3])
                .collect();

            msh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        }
    }
}
