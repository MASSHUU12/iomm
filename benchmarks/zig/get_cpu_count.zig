const std = @import("std");

const c = @cImport({
    @cInclude("unistd.h");
});

pub fn getCpuCount() usize {
    // POSIX
    return @intCast(c.sysconf(c._SC_NPROCESSORS_ONLN));
}
