// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"context"
	"log"

	"github.com/nlpodyssey/spago/cmd/clientutils"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"github.com/urfave/cli"
)

func newClientClassifyCommandFor(app *BertApp) cli.Command {
	return cli.Command{
		Name:        "classify",
		Usage:       "Perform text classification using BERT.",
		Description: "Run the " + programName + " client for text classification.",
		Flags:       newClientClassifyCommandFlagsFor(app),
		Action:      newClientClassifyCommandActionFor(app),
	}
}

func newClientClassifyCommandFlagsFor(app *BertApp) []cli.Flag {
	return clientutils.Flags(&app.address, &app.tlsDisable, &app.output, []cli.Flag{
		cli.StringFlag{
			Name:        "text",
			Destination: &app.requestText,
			Required:    true,
		},
	})
}

func newClientClassifyCommandActionFor(app *BertApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		clientutils.VerifyFlags(app.output)

		conn := clientutils.OpenConnection(app.address, app.tlsDisable)
		client := grpcapi.NewBERTClient(conn)

		resp, err := client.Classify(context.Background(), &grpcapi.ClassifyRequest{
			Text: app.requestText,
		})

		if err != nil {
			log.Fatalln(err)
		}

		clientutils.Println(app.output, resp)
	}
}
