def write_videoclip(clip, path, fps=None, n_thread=4):
    getattr(clip, "write_videofile")(
        path,
        fps=fps,
        codec="libx264",
        threads=n_thread,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
