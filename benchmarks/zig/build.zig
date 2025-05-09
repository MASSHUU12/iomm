const std = @import("std");

pub fn build(b: *std.Build) !void {
    const mode = b.standardOptimizeOption(.{});
    const target_options = b.standardTargetOptions(.{});
    const cwd_handle = std.fs.cwd();

    var dir = try cwd_handle.openDir(".", .{ .iterate = true });
    defer dir.close();

    var it = dir.iterate();
    while (true) {
        const entry = try it.next() orelse break;
        const name = entry.name;
        if (std.mem.eql(u8, name, "build.zig") or !std.mem.endsWith(u8, name, ".zig")) continue;

        // strip “.zig” to get executable name
        const exe_name = name[0 .. name.len - 4];
        const exe = b.addExecutable(.{ .name = exe_name, .target = target_options, .root_source_file = b.path(name), .optimize = mode });
        // install into zig-out/bin
        b.installArtifact(exe);
    }
}
