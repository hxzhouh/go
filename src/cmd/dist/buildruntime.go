// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"strings"
)

/*
 * Helpers for building runtime.
 */

// mkzversion writes zversion.go:
//
//	package sys
//
// (Nothing right now!)
func mkzversion(dir, file string) {
	var buf strings.Builder
	writeHeader(&buf)
	fmt.Fprintf(&buf, "package sys\n")
	writefile(buf.String(), file, writeSkipSame)
}

// mkbuildcfg writes internal/buildcfg/zbootstrap.go:
//
//	package buildcfg
//
//	const defaultGOROOT = <goroot>
//	const defaultGO386 = <go386>
//	...
//	const defaultGOOS = runtime.GOOS
//	const defaultGOARCH = runtime.GOARCH
//
// The use of runtime.GOOS and runtime.GOARCH makes sure that
// a cross-compiled compiler expects to compile for its own target
// system. That is, if on a Mac you do:
//
//	GOOS=linux GOARCH=ppc64 go build cmd/compile
//
// the resulting compiler will default to generating linux/ppc64 object files.
// This is more useful than having it default to generating objects for the
// original target (in this example, a Mac).
func mkbuildcfg(file string) {
	var buf strings.Builder
	writeHeader(&buf)
	fmt.Fprintf(&buf, "package buildcfg\n")
	fmt.Fprintln(&buf)
	fmt.Fprintf(&buf, "import \"runtime\"\n")
	fmt.Fprintln(&buf)
	fmt.Fprintf(&buf, "const DefaultGO386 = `%s`\n", go386)
	fmt.Fprintf(&buf, "const DefaultGOAMD64 = `%s`\n", goamd64)
	fmt.Fprintf(&buf, "const DefaultGOARM = `%s`\n", goarm)
	fmt.Fprintf(&buf, "const DefaultGOARM64 = `%s`\n", goarm64)
	fmt.Fprintf(&buf, "const DefaultGOMIPS = `%s`\n", gomips)
	fmt.Fprintf(&buf, "const DefaultGOMIPS64 = `%s`\n", gomips64)
	fmt.Fprintf(&buf, "const DefaultGOPPC64 = `%s`\n", goppc64)
	fmt.Fprintf(&buf, "const DefaultGORISCV64 = `%s`\n", goriscv64)
	fmt.Fprintf(&buf, "const defaultGOEXPERIMENT = `%s`\n", goexperiment)
	fmt.Fprintf(&buf, "const defaultGO_EXTLINK_ENABLED = `%s`\n", goextlinkenabled)
	fmt.Fprintf(&buf, "const defaultGO_LDSO = `%s`\n", defaultldso)
	fmt.Fprintf(&buf, "const version = `%s`\n", findgoversion())
	fmt.Fprintf(&buf, "const defaultGOOS = runtime.GOOS\n")
	fmt.Fprintf(&buf, "const defaultGOARCH = runtime.GOARCH\n")
	fmt.Fprintf(&buf, "const DefaultGOFIPS140 = `%s`\n", gofips140)
	fmt.Fprintf(&buf, "const DefaultCGO_ENABLED = %s\n", quote(os.Getenv("CGO_ENABLED")))

	writefile(buf.String(), file, writeSkipSame)
}

// mkobjabi writes cmd/internal/objabi/zbootstrap.go:
//
//	package objabi
//
// (Nothing right now!)
func mkobjabi(file string) {
	var buf strings.Builder
	writeHeader(&buf)
	fmt.Fprintf(&buf, "package objabi\n")

	writefile(buf.String(), file, writeSkipSame)
}
